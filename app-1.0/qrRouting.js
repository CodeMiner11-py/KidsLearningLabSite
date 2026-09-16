// qrRouting.js — Kids Learning Lab: "what does this scanned code mean?"
//
// The app has exactly two kinds of QR payload in circulation:
//   - a short 4-character game join code (multiplayer.js's randomCode()),
//     which lives in one of the Realtime Database namespaces in
//     GAME_SESSION_PATHS below
//   - a person's raw Firebase uid — their "My QR Code" from the Friends
//     page (main.js) — which resolves via the Firestore `userDirectory`
//     collection
//
// Three different screens can open the camera (Friends page's "Scan to
// Add", Learn's "Share Course" scanner, and "Join a Game"'s scanner), but
// whichever one actually gets pointed at a code, the result should be the
// same: a game code always opens that game, a person code always resolves
// to that person — never "sends a friend request to a 4-character game
// code" or "fails to join a game because you happened to scan it from the
// wrong screen". identifyScannedCode() is the one place that inspects a
// scanned string and decides which of those two things it is, so main.js
// and learn.js don't each reimplement (and risk disagreeing on) that call.
//
// Deliberately its own module:
//   - not folded into multiplayer.js, which stays game-agnostic and knows
//     nothing about Firestore or user accounts
//   - not folded into main.js or learn.js, which would create a circular
//     import between them (main.js already imports learn.js)
import { db } from './firebase.js';
import { doc, getDoc } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";
import { sessionExists } from './multiplayer.js';

// Every game with a remote mode gets its own RTDB namespace. This is the
// single source of truth for that list — learn.js imports it from here
// (rather than each keeping its own copy) so the two can't drift apart.
export const GAME_SESSION_PATHS = ['seesawSessions', 'duelSessions'];

// Game codes are always exactly 4 characters from multiplayer.js's
// unambiguous charset (uppercase letters/digits, no 0/O or 1/I). A real
// Firebase uid landing in that exact shape is astronomically unlikely, so
// this is a cheap way to decide whether a network round-trip to check "is
// this a live game code" is even worth making.
const GAME_CODE_SHAPE = /^[ABCDEFGHJKLMNPQRSTUVWXYZ23456789]{4}$/;

// Inspects a scanned (or typed) string and figures out what it actually
// is. Always trims but never uppercases — callers must scan with
// preserveCase: true (see multiplayer.js's scanJoinCode) since a person
// code is case-sensitive; game codes happen to already be all-uppercase so
// preserving case never breaks that half.
//   { type: 'game', gamePath, code }   — a live game session code
//   { type: 'person', uid }            — a real account's uid
//   { type: 'unknown', code }          — neither (bad, stale, or garbled)
export async function identifyScannedCode(rawText) {
  const code = String(rawText || '').trim();
  if (!code) return { type: 'unknown', code };

  if (GAME_CODE_SHAPE.test(code)) {
    for (const gamePath of GAME_SESSION_PATHS) {
      try {
        if (await sessionExists(gamePath, code)) return { type: 'game', gamePath, code };
      } catch { /* RTDB hiccup on this namespace — try the next one */ }
    }
  }

  try {
    const dirSnap = await getDoc(doc(db, 'userDirectory', code));
    if (dirSnap.exists()) return { type: 'person', uid: code };
  } catch { /* not shaped like a real uid, or rules rejected the read */ }

  return { type: 'unknown', code };
}