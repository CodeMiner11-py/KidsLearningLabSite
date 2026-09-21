// gameHistory.js — Kids Learning Lab Learning Games history
//
// Every game (trivia, maze, seesaw, duel, meltdown, wordGrid, connectors)
// keeps its own running history of generated words/questions in Firestore,
// under users/{uid}/learnProfile/main/gameHistory/{gameId}. That history is:
//   1. Recorded once the learner actually presses Start (gateAndStartGame in
//      learn.js calls recordGameHistory right before launching the game) —
//      NOT at generate time, so a generated-but-abandoned set never counts.
//   2. Read back before every new generation (getGameHistoryContext) and
//      sent to the Learn Worker as `history: [{ text, daysAgo }, ...]` so the
//      AI prompt can steer away from anything played recently, while still
//      allowing older items back in if it's running out of fresh material.
//   3. Consumed by the Learn Worker's historyContextBlock() helper, which
//      splits it into "recently played — do not reuse" vs. "played a while
//      ago — OK as a last resort" for the prompt.
//
import { auth, db } from './firebase.js';
import {
  doc, getDoc, setDoc
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";

// Kept small enough to stay well under Firestore's 1MB doc limit even for a
// very active player, while still giving the AI a decent memory window.
const MAX_STORED_ITEMS = 25;    // trimmed oldest-first on every write
const MAX_CONTEXT_ITEMS = 25;   // most-recent slice actually sent to the AI (can't exceed what's stored anyway)

// Local mirror of the same history. Firestore stays the main copy, but every
// read/write below already fails quietly (caught, then ignored), so a
// security-rules rejection or a flaky connection used to mean history was
// always empty and games repeated. The local copy means it still remembers.
function localKey(gameId) {
  const u = auth.currentUser;
  return u ? `kll_game_history_${u.uid}_${gameId}` : null;
}
function readLocal(gameId) {
  try {
    const k = localKey(gameId);
    const parsed = k ? JSON.parse(localStorage.getItem(k) || '[]') : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch { return []; }
}
function writeLocal(gameId, items) {
  try {
    const k = localKey(gameId);
    if (k) localStorage.setItem(k, JSON.stringify(items));
  } catch { /* storage full or unavailable */ }
}
// Union of two item lists, de-duplicated, oldest first.
function mergeItems(a, b) {
  const seen = new Set();
  return [...a, ...b]
    .filter((i) => i && i.text && typeof i.playedAt === 'number')
    .filter((i) => {
      const k = `${i.text}|${i.playedAt}`;
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    })
    .sort((x, y) => x.playedAt - y.playedAt);
}

function historyDocRef(gameId) {
  const u = auth.currentUser;
  if (!u) return null;
  return doc(db, 'users', u.uid, 'learnProfile', 'main', 'gameHistory', gameId);
}

// Call once the learner actually starts a game (see gateAndStartGame in
// learn.js) — records every word/question text from `texts` with "now" as
// the played timestamp. Fire-and-forget from the caller's perspective: never
// throws, never blocks the game from starting.
export async function recordGameHistory(gameId, texts) {
  const ref = historyDocRef(gameId);
  const clean = (texts || []).filter(Boolean).map((t) => String(t).trim()).filter(Boolean);
  if (!ref || !clean.length) return;

  const now = Date.now();
  const additions = clean.map((text) => ({ text, playedAt: now }));

  // Local first, synchronously, so the very next generation sees it even if
  // the Firestore round trip below is slow or fails.
  const local = mergeItems(readLocal(gameId), additions).slice(-MAX_STORED_ITEMS);
  writeLocal(gameId, local);

  try {
    const snap = await getDoc(ref);
    const existing = snap.exists() ? (snap.data().items || []) : [];
    const merged = mergeItems(existing, local).slice(-MAX_STORED_ITEMS);
    writeLocal(gameId, merged);
    await setDoc(ref, { items: merged }, { merge: true });
  } catch (err) {
    // console.error, not warn: if this shows up, Firestore rules for
    // users/{uid}/learnProfile/main/gameHistory/{gameId} are the first suspect.
    console.error(`Failed to save ${gameId} history to Firestore (kept locally):`, err);
  }
}

// Call right before asking the worker to generate a new set. Returns
// [{ text, daysAgo }, ...], most-recently-played first, capped at
// MAX_CONTEXT_ITEMS — ready to drop straight into the worker request body
// so it can tell fresh material from stuff the learner has already seen.
export async function getGameHistoryContext(gameId) {
  const ref = historyDocRef(gameId);
  if (!ref) return [];

  let remote = [];
  try {
    const snap = await getDoc(ref);
    remote = snap.exists() ? (snap.data().items || []) : [];
  } catch (err) {
    console.error(`Failed to load ${gameId} history from Firestore (using local copy):`, err);
  }

  {
    const items = mergeItems(remote, readLocal(gameId));
    const now = Date.now();
    return items
      .slice(-MAX_CONTEXT_ITEMS)
      .reverse()
      .map(({ text, playedAt }) => ({
        text,
        daysAgo: Math.max(0, Math.floor((now - playedAt) / 86400000)),
      }));
  }
}