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

  try {
    const snap = await getDoc(ref);
    const existing = snap.exists() ? (snap.data().items || []) : [];
    const now = Date.now();
    const additions = clean.map((text) => ({ text, playedAt: now }));
    const merged = existing.concat(additions).slice(-MAX_STORED_ITEMS);
    await setDoc(ref, { items: merged }, { merge: true });
  } catch (err) {
    console.warn(`Failed to record ${gameId} history:`, err);
  }
}

// Call right before asking the worker to generate a new set. Returns
// [{ text, daysAgo }, ...], most-recently-played first, capped at
// MAX_CONTEXT_ITEMS — ready to drop straight into the worker request body
// so it can tell fresh material from stuff the learner has already seen.
export async function getGameHistoryContext(gameId) {
  const ref = historyDocRef(gameId);
  if (!ref) return [];

  try {
    const snap = await getDoc(ref);
    if (!snap.exists()) return [];
    const items = snap.data().items || [];
    const now = Date.now();
    return items
      .slice(-MAX_CONTEXT_ITEMS)
      .reverse()
      .map(({ text, playedAt }) => ({
        text,
        daysAgo: Math.max(0, Math.floor((now - playedAt) / 86400000)),
      }));
  } catch (err) {
    console.warn(`Failed to load ${gameId} history:`, err);
    return [];
  }
}