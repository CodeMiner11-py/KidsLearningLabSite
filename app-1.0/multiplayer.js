// multiplayer.js — Kids Learning Lab cross-device multiplayer sessions
//
// Generic Realtime Database session helper used by any "Play With a Friend"
// game mode (currently Seesaw). Deliberately game-agnostic: callers pass a
// `gamePath` (e.g. "seesawSessions") so each game gets its own RTDB
// namespace, plus whatever shape of data that game needs — this file just
// handles creating a joinable session behind a short code, joining one,
// listening for changes, patching it, and tearing it down.
//
// Uses the Firebase Realtime Database, NOT Firestore — RTDB's onValue
// listeners give sub-second sync for turn-by-turn game state, which is what
// this needs. Firestore (via firebase.js) is left alone for everything else.
//
// Requires Realtime Database to be enabled for the Firebase project. If the
// app was initialized (in firebase.js) with a config that doesn't include a
// `databaseURL`, getDatabase() below will throw — see the note at the
// bottom of this file for the one-line fix.
import { getApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import {
  getDatabase, ref, get, set, update, remove,
  onValue, off, onDisconnect, runTransaction,
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-database.js";
// QR generation only — pure JS, no native dependency. Pinned version, ESM
// build served by jsdelivr (same CDN-import pattern as the firebase SDK
// above), so no npm/bundler step is needed for this half of the feature.
import QRCode from "https://cdn.jsdelivr.net/npm/qrcode@1.5.4/+esm";
// QR scanning, also pure JS — decodes frames from a plain getUserMedia
// <video> stream in the webview itself. No Capacitor plugin, no native
// build step at all: this is why it replaced the earlier
// @capacitor/barcode-scanner approach, which hit a Capacitor 8 + SPM
// linking bug (plugin synced but never actually exposed to Xcode).
// v1.4.2 lazy-loads its own decode worker via a dynamic import relative to
// this URL, so nothing else needs to be configured for the CDN build.
import QrScanner from "https://cdn.jsdelivr.net/npm/qr-scanner@1.4.2/qr-scanner.min.js";

let _rtdb = null;
function rtdb() {
  if (!_rtdb) _rtdb = getDatabase(getApp());
  return _rtdb;
}

// Unambiguous uppercase charset — no 0/O or 1/I, so a friend reading a code
// off a screen never has to guess which letter/number they're looking at.
const CODE_CHARS = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
function randomCode(len = 4) {
  let s = '';
  for (let i = 0; i < len; i++) s += CODE_CHARS[Math.floor(Math.random() * CODE_CHARS.length)];
  return s;
}

// Creates a new session at `${gamePath}/${code}` with a freshly-generated
// code, retrying on the rare collision. `initialData` is merged with
// { status: 'waiting', createdAt }. Registers an onDisconnect so an
// abandoned host (app killed, phone locked, network drop) doesn't leave a
// friend staring at a code that will never connect — the session flips to
// 'ended' automatically. Returns the code on success.
export async function hostSession(gamePath, initialData) {
  const database = rtdb();
  for (let attempt = 0; attempt < 6; attempt++) {
    const code = randomCode(4);
    const sessionRef = ref(database, `${gamePath}/${code}`);
    const result = await runTransaction(sessionRef, (current) => {
      if (current !== null) return; // code taken — abort this attempt, try another
      return { ...initialData, status: 'waiting', createdAt: Date.now() };
    });
    if (result.committed) {
      onDisconnect(sessionRef).update({ status: 'ended', endedReason: 'host_left' });
      return code;
    }
  }
  throw new Error('Could not create a game code — try again.');
}

// Joins an existing waiting session by code. `guestData` is merged in along
// with status: 'active'. Throws a friendly error if the code doesn't exist,
// already started, or already ended. Returns { code, session } where
// `session` is the merged post-join state.
export async function joinSession(gamePath, code, guestData) {
  const database = rtdb();
  const cleanCode = String(code || '').trim().toUpperCase();
  if (!cleanCode) throw new Error('Type a code first.');
  const sessionRef = ref(database, `${gamePath}/${cleanCode}`);
  const snap = await get(sessionRef);
  if (!snap.exists()) throw new Error("That code wasn't found. Check it and try again.");
  const data = snap.val();
  if (data.status === 'ended') throw new Error('That game has ended.');
  if (data.status !== 'waiting') throw new Error('That game already started without you.');

  const merged = { ...guestData, status: 'active', activeAt: Date.now() };
  await update(sessionRef, merged);
  onDisconnect(sessionRef).update({ status: 'ended', endedReason: 'guest_left' });
  return { code: cleanCode, session: { ...data, ...merged } };
}

// Subscribes to a session, calling `callback(sessionOrNull)` immediately
// with the current value and again on every change. Returns an unsubscribe
// function — always call it when leaving the game or the listener leaks.
export function listenToSession(gamePath, code, callback) {
  const database = rtdb();
  const sessionRef = ref(database, `${gamePath}/${code}`);
  const handler = (snap) => callback(snap.exists() ? snap.val() : null);
  onValue(sessionRef, handler);
  return () => off(sessionRef, 'value', handler);
}

// Shallow-merges `patch` into the session — the workhorse for turn-by-turn
// updates (whose turn it is, the current question, scores, timer anchor).
export async function updateSession(gamePath, code, patch) {
  const database = rtdb();
  await update(ref(database, `${gamePath}/${code}`), patch);
}

// Deletes a session outright — used when a host cancels before anyone joins.
export async function cancelSession(gamePath, code) {
  const database = rtdb();
  await remove(ref(database, `${gamePath}/${code}`));
}

// Read-only existence check — does NOT join, doesn't touch status or
// register onDisconnect. Used to figure out whether a scanned/typed code
// is actually a live game session before committing to either that or some
// other interpretation of the code (see qrRouting.js), without the side
// effects joinSession() has.
export async function sessionExists(gamePath, code) {
  const database = rtdb();
  const snap = await get(ref(database, `${gamePath}/${code}`));
  return snap.exists();
}

// Atomically claims a field at `${gamePath}/${code}/${field}` — used for
// "first correct answer wins" race resolution (e.g. Duel), where both
// devices may try to report a correct answer for the same round at nearly
// the same instant. Only the first caller to run the transaction while the
// field is still empty gets `committed: true`; everyone else's attempt is
// aborted and sees `committed: false`, so exactly one device ends up
// responsible for writing the round's resolution (score bump, next round).
// `field` may be a nested path, e.g. `roundWinners/3`.
export async function claimField(gamePath, code, field, value) {
  const database = rtdb();
  const fieldRef = ref(database, `${gamePath}/${code}/${field}`);
  const result = await runTransaction(fieldRef, (current) => {
    if (current !== null && current !== undefined) return; // already claimed — abort
    return value;
  });
  return result.committed;
}

// ---------------------------------------------------------------------
// QR code join — scan-to-join for "Play With a Friend" sessions.
//
// The QR encodes ONLY the bare 4-character session code (e.g. "H7K2"), the
// same string a host would read aloud or text to a friend. Deliberately NOT
// a URL or deep link: a plain text payload means the system Camera app has
// nothing actionable to open (no Safari prompt), and iOS just shows inert
// text — so the code can only meaningfully be redeemed by scanning inside
// Kids Learning Lab's own "Join a Game" flow. Which game the code belongs
// to is resolved the same way manual entry already resolves it — by trying
// each namespace in GAME_SESSION_PATHS (see joinAnyGameSession in learn.js)
// — so the QR payload can stay this simple.
// ---------------------------------------------------------------------

// Renders `code` as a QR code onto an existing <canvas> element (sized by
// the canvas's own width/height attributes — see the CSS/HTML side for
// sizing). Safe to call again on the same canvas to redraw (e.g. once the
// real code replaces a "····" placeholder).
export async function renderJoinQr(canvasEl, code) {
  if (!canvasEl || !code) return;
  await QRCode.toCanvas(canvasEl, code, {
    width: canvasEl.width || 160,
    margin: 1,
    color: { dark: '#123B7A', light: '#FFFFFF' },
  });
}

// Opens a full-screen in-webview camera scanner (markup lives in
// index.html as #qrScanOverlay / #qrScanVideo / #qrScanStatus /
// #qrScanError / #qrScanCancelBtn) and resolves to the scanned text,
// trimmed the same way a typed code is. Resolves null if the user taps
// Cancel. Rejects with a friendly Error if the camera can't be opened at
// all (permission denied, no camera) — callers are responsible for turning
// that into a message next to whatever manual-entry UI they have, same as
// any other join failure. Only one scan can be in flight at a time.
//
// Shared by every "scan a code" flow in the app — game join codes (short,
// uppercase-only), and friend/course-share codes (a Firebase uid, which is
// mixed-case and must NOT be uppercased or it stops matching). Options:
//   statusText   — overrides the default "friend's code" prompt so the
//                  overlay can say something specific ("friend's QR code",
//                  "the course code", etc).
//   preserveCase — when true, the scanned text is trimmed only, not
//                  uppercased. Game codes want the default (false); uid
//                  payloads must pass true.
export async function scanJoinCode({ statusText, preserveCase = false } = {}) {
  const overlay = document.getElementById('qrScanOverlay');
  const video = document.getElementById('qrScanVideo');
  const statusEl = document.getElementById('qrScanStatus');
  const errorEl = document.getElementById('qrScanError');
  const cancelBtn = document.getElementById('qrScanCancelBtn');
  if (!overlay || !video || !cancelBtn) throw new Error('Scanner UI is missing from this page.');

  errorEl.textContent = '';
  statusEl.textContent = statusText || "Point the camera at your friend's code";
  overlay.classList.add('show');

  return new Promise((resolve, reject) => {
    let settled = false;
    let scanner = null;

    const finish = (value, err) => {
      if (settled) return;
      settled = true;
      cancelBtn.removeEventListener('click', onCancel);
      try { scanner?.stop(); } catch { /* already stopped */ }
      try { scanner?.destroy(); } catch { /* already destroyed */ }
      overlay.classList.remove('show');
      if (err) reject(err); else resolve(value);
    };
    const onCancel = () => finish(null);
    cancelBtn.addEventListener('click', onCancel);

    scanner = new QrScanner(
      video,
      (result) => {
        let text = String(result?.data ?? result ?? '').trim();
        if (!preserveCase) text = text.toUpperCase();
        if (text) finish(text);
      },
      {
        onDecodeError: () => {}, // fires continuously while no code is in frame — expected, not a real error
        highlightScanRegion: false,
        highlightCodeOutline: false,
        preferredCamera: 'environment',
      }
    );

    scanner.start().catch((err) => {
      const reason = err?.message || err?.name || '';
      const message = /(permission|denied|notallowed)/i.test(reason)
        ? 'Camera access was denied. Enable it in Settings to scan codes.'
        : 'Could not open the camera.';
      finish(null, new Error(message));
    });
  });
}

// ---------------------------------------------------------------------
// One-time setup note: getApp()/getDatabase() reuse whatever Firebase app
// firebase.js already initialized (via initializeApp) for Auth/Firestore —
// this file never calls initializeApp itself. The only thing this project
// needs for multiplayer to work is Realtime Database turned on:
//   1. Firebase Console → Build → Realtime Database → Create Database.
//   2. Set rules so a signed-in user can read/write session docs, e.g.:
//        {
//          "rules": {
//            "seesawSessions": {
//              "$code": { ".read": "auth != null", ".write": "auth != null" }
//            }
//          }
//        }
//   3. If initializeApp's config in firebase.js was copied before Realtime
//      Database existed, it may be missing `databaseURL` — grab it from
//      Console → Realtime Database → Data tab (top of the page) and add it
//      to that config object. If it's already there, nothing else to do.
// ---------------------------------------------------------------------