// groupGame.js — Kids Learning Lab "Group Game" (2 Minutes) data layer
//
// Unlike multiplayer.js's 1v1 sessions (Seesaw/Duel — two fixed player
// slots, one shared question queue), a Group Game "room" holds up to 6
// independent players who each play their OWN local instance of a
// familiar solo game (Maze, Meltdown, Word Grid, or Word Connectors) on
// the same topic at the same time, racing to rack up correct answers
// before a shared 2-minute clock runs out. The room in Realtime Database
// exists to sync: who's in the lobby, which game/topic the host picked,
// when the shared clock started, and a live per-player score — never the
// questions themselves (each client generates/refetches its own set via
// the same Learn Worker calls the solo games already use).
//
// Lives at `groupRooms/{code}` in RTDB — its own namespace, separate from
// multiplayer.js's `seesawSessions`/`duelSessions`, so a 4-char code can
// never collide across the two systems. Not folded into multiplayer.js
// because the shape here (a growable player map, host-controlled round
// transitions, additive score reporting) doesn't fit that file's
// two-fixed-slot session model.
import { getApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import {
  getDatabase, ref, get, set, update, remove,
  onValue, off, onDisconnect, runTransaction,
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-database.js";

let _rtdb = null;
function rtdb() {
  if (!_rtdb) _rtdb = getDatabase(getApp());
  return _rtdb;
}

// Same unambiguous charset multiplayer.js uses for game codes — kept as a
// local copy since multiplayer.js doesn't export its randomCode() helper,
// and duplicating one tiny function beats importing across an unrelated
// concern.
const CODE_CHARS = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
function randomCode(len = 4) {
  let s = '';
  for (let i = 0; i < len; i++) s += CODE_CHARS[Math.floor(Math.random() * CODE_CHARS.length)];
  return s;
}

export const MAX_PLAYERS = 6;

// Six hand-picked, maximally distinguishable colors — assigned to players
// in join order so the lobby roster and live scoreboard HUD can tell
// everyone apart at a glance without relying on names alone.
export const PLAYER_COLORS = [
  '#1E6FE0', // blue
  '#E0503A', // red-orange
  '#2FAE66', // green
  '#F2B705', // yellow
  '#9B59D9', // purple
  '#FF7CB0', // pink
];

const DEFAULT_ROUND_MS = 120000; // 2 minutes

function roomRef(code) {
  return ref(rtdb(), `groupRooms/${code}`);
}

// Creates a new room in 'lobby' status with the host as its first player
// (assigned PLAYER_COLORS[0]). Retries on the rare code collision, same
// pattern as multiplayer.js's hostSession. Registers an onDisconnect so a
// host who closes the app mid-lobby doesn't leave guests staring at a room
// that will never start. Returns the code on success.
export async function createRoom(hostUid, hostName) {
  const database = rtdb();
  for (let attempt = 0; attempt < 6; attempt++) {
    const code = randomCode(4);
    const ref_ = roomRef(code);
    const result = await runTransaction(ref_, (current) => {
      if (current !== null) return; // code taken — try another
      return {
        status: 'lobby',
        hostUid,
        hostName,
        createdAt: Date.now(),
        gameId: null,
        topic: null,
        difficulty: null,
        roundStartedAt: null,
        roundDurationMs: DEFAULT_ROUND_MS,
        players: {
          [hostUid]: { name: hostName, color: PLAYER_COLORS[0], score: 0, joinedAt: Date.now(), status: 'lobby' },
        },
      };
    });
    if (result.committed) {
      onDisconnect(ref_).update({ status: 'ended', endedReason: 'host_left' });
      return code;
    }
  }
  throw new Error('Could not create a room code — try again.');
}

// Joins an existing lobby by code. Fails if the room doesn't exist, has
// already started/ended, or is full. Colors are assigned via a transaction
// on the whole room so two guests joining in the same instant can't both
// grab the same color or push the roster past MAX_PLAYERS. Returns
// { code, room } — the post-join room state.
export async function joinRoom(code, uid, name) {
  const database = rtdb();
  const cleanCode = String(code || '').trim().toUpperCase();
  if (!cleanCode) throw new Error('Type a code first.');
  const ref_ = roomRef(cleanCode);

  const result = await runTransaction(ref_, (room) => {
    if (room === null) return room; // abort — handled below as "not found"
    if (room.status === 'ended') return; // abort
    if (room.status !== 'lobby') return; // abort — already started
    const players = room.players || {};
    if (players[uid]) return room; // already in — no-op, just re-join the listener
    if (Object.keys(players).length >= MAX_PLAYERS) return; // abort — full

    const usedColors = new Set(Object.values(players).map((p) => p.color));
    const color = PLAYER_COLORS.find((c) => !usedColors.has(c)) || PLAYER_COLORS[Object.keys(players).length % PLAYER_COLORS.length];
    players[uid] = { name, color, score: 0, joinedAt: Date.now(), status: 'lobby' };
    room.players = players;
    return room;
  });

  if (!result.committed || !result.snapshot.exists()) {
    // Distinguish "never existed" from "existed but transaction aborted"
    // for a clearer message.
    const snap = await get(ref_);
    if (!snap.exists()) throw new Error("That code wasn't found. Check it and try again.");
    const room = snap.val();
    if (room.status === 'ended') throw new Error('That group has ended.');
    if (room.status !== 'lobby') throw new Error('That group already started without you.');
    throw new Error("That group is full — six players is the max.");
  }

  onDisconnect(ref(database, `groupRooms/${cleanCode}/players/${uid}`)).remove();
  return { code: cleanCode, room: result.snapshot.val() };
}

// Subscribes to a room, calling `callback(roomOrNull)` immediately with the
// current value and again on every change. Returns an unsubscribe function.
export function listenToRoom(code, callback) {
  const database = rtdb();
  const ref_ = ref(database, `groupRooms/${code}`);
  const handler = (snap) => callback(snap.exists() ? snap.val() : null);
  onValue(ref_, handler);
  return () => off(ref_, 'value', handler);
}

// Host-only: kicks off a round — writes the chosen game/topic/difficulty,
// flips status to 'playing', stamps roundStartedAt (every client computes
// its own local countdown from this + roundDurationMs, so no per-second
// RTDB writes are needed to keep the shared clock in sync), and resets
// every player's score to 0 for the new round.
export async function startRound(code, { gameId, topic, difficulty, roundDurationMs = DEFAULT_ROUND_MS }) {
  const database = rtdb();
  const snap = await get(roomRef(code));
  if (!snap.exists()) throw new Error('That group no longer exists.');
  const room = snap.val();
  const resetPlayers = {};
  for (const uid of Object.keys(room.players || {})) {
    resetPlayers[uid] = { ...room.players[uid], score: 0, status: 'playing' };
  }
  await update(roomRef(code), {
    status: 'playing',
    gameId,
    topic,
    difficulty,
    roundStartedAt: Date.now(),
    roundDurationMs,
    players: resetPlayers,
  });
}

// Atomically bumps one player's score by 1 — called every time that
// player's local game registers a correct answer / solved group / round
// win. A transaction (rather than a plain read-then-write) so rapid-fire
// correct answers from a fast player never clobber each other.
export async function reportCorrectAnswer(code, uid) {
  const database = rtdb();
  const scoreRef = ref(database, `groupRooms/${code}/players/${uid}/score`);
  await runTransaction(scoreRef, (current) => (current || 0) + 1);
}

// Marks a player as having finished their local round early (e.g. they
// closed the mini-game instead of playing until the clock ran out). Purely
// cosmetic for the live HUD/results screen — never blocks other players.
export async function markPlayerDone(code, uid) {
  const database = rtdb();
  await update(ref(database, `groupRooms/${code}/players/${uid}`), { status: 'done' });
}

// Host-only: flips the room to 'results' once the shared clock has run
// out. Harmless if called more than once (e.g. a slightly-late client) —
// just overwrites status with the same value.
export async function endRound(code) {
  await update(roomRef(code), { status: 'results' });
}

// Host-only: sends everyone back to the game/topic picker for another
// round, keeping the same roster. Scores are left as-is until the next
// startRound() reset so the just-finished results screen still reads
// correctly for anyone who round-trips slowly.
export async function resetRoomForNewRound(code) {
  await update(roomRef(code), {
    status: 'picking', gameId: null, topic: null, difficulty: null, roundStartedAt: null,
  });
}

// Removes one player from the room. If the leaving player is the host,
// the whole room is ended instead — there's no "reassign host" flow, and
// leaving a headless room running would strand everyone else.
export async function leaveRoom(code, uid) {
  const database = rtdb();
  const snap = await get(roomRef(code));
  if (!snap.exists()) return;
  const room = snap.val();
  if (room.hostUid === uid) {
    await update(roomRef(code), { status: 'ended', endedReason: 'host_left' });
  } else {
    await remove(ref(database, `groupRooms/${code}/players/${uid}`));
  }
}

// Deletes a room outright — used when a host cancels from the lobby before
// starting anything.
export async function cancelRoom(code) {
  const database = rtdb();
  await remove(roomRef(code));
}