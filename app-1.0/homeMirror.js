// homeMirror.js — background Firestore → RTDB mirror for Home screen data
//
// Firestore stays the source of truth for everything here (learnProfile,
// learnCourses, friends, sharedCourses, wrongAnswers) — this module doesn't
// change how or where any of that gets WRITTEN. It only listens for changes
// with onSnapshot (already the established pattern in this app — see
// startXpListener/startNotifBell in main.js) and mirrors the current value
// into RTDB under homeData/{uid}/... in the background, so loadHomeData()
// in main.js can read from RTDB instead of doing several Firestore reads
// (including a couple of query()s) every time Home refreshes.
//
// Listening instead of dual-writing at each of the many scattered write
// sites (lesson completion, friend accept/reject, course creation, XP
// awards, etc.) means nothing can update Firestore without the mirror
// picking it up automatically — no risk of a future write site being
// added and forgotten.
//
// Every mirror write is fire-and-forget (.catch-only, never awaited by a
// caller) — a slow or failed RTDB write must never block or delay the
// actual user-facing action that triggered the underlying Firestore
// change. Worst case of a missed/delayed mirror write is loadHomeData()
// briefly reading a stale RTDB copy; loadHomeData() below still has its
// own Firestore fallback for exactly that reason.
import { auth, db, rtdb } from './firebase.js';
import {
  doc, collection, query, where, orderBy, limit, onSnapshot
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";
import {
  ref as rtdbRef, set as rtdbSet, update as rtdbUpdate
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-database.js";

let _unsubs = [];
let _unsubWrongAnswers = null;
let _wrongAnswersCourseId = null;

// Firestore Timestamp objects (e.g. lastOpenedAt, written via
// serverTimestamp()) aren't plain-JSON-serializable and rtdbSet()/rtdbUpdate()
// will throw on them. Recursively convert any Timestamp-shaped value (has a
// toMillis() method — true for Firestore Timestamp instances) to a plain
// epoch-ms number before writing to RTDB. Leaves everything else untouched.
function sanitizeForRtdb(value) {
  if (value === null || value === undefined) return value;
  if (typeof value?.toMillis === 'function') return value.toMillis();
  if (Array.isArray(value)) return value.map(sanitizeForRtdb);
  if (typeof value === 'object') {
    const out = {};
    for (const [k, v] of Object.entries(value)) out[k] = sanitizeForRtdb(v);
    return out;
  }
  return value;
}

function mirrorPath(uid, ...segments) {
  return rtdbRef(rtdb, ['homeData', uid, ...segments].join('/'));
}

function bgSet(refPath, value) {
  rtdbSet(refPath, sanitizeForRtdb(value)).catch((err) => {
    console.warn('homeMirror: background RTDB write failed for', refPath.toString(), err);
  });
}

function bgUpdate(refPath, value) {
  rtdbUpdate(refPath, sanitizeForRtdb(value)).catch((err) => {
    console.warn('homeMirror: background RTDB update failed for', refPath.toString(), err);
  });
}

// ---- learnProfile/main → homeData/{uid}/learnProfile ----
// Only field loadHomeData() actually reads off this doc is xp, but we
// mirror the whole doc rather than hand-picking fields — cheap, and means
// this mirror never silently drifts if loadHomeData() starts reading
// something else off this doc later.
function watchLearnProfile(uid) {
  const q = doc(db, 'users', uid, 'learnProfile', 'main');
  return onSnapshot(q, (snap) => {
    bgSet(mirrorPath(uid, 'learnProfile'), snap.exists() ? snap.data() : { xp: 0 });
  }, (err) => console.warn('homeMirror: learnProfile listener error:', err));
}

// ---- learnCourses (most-recently-opened) → homeData/{uid}/activeCourse ----
// Same query loadHomeData() itself runs today: top 1 by lastOpenedAt desc.
function watchActiveCourse(uid) {
  const q = query(collection(db, 'users', uid, 'learnCourses'), orderBy('lastOpenedAt', 'desc'), limit(1));
  return onSnapshot(q, (snap) => {
    const activeCourse = !snap.empty ? { id: snap.docs[0].id, ...snap.docs[0].data() } : null;
    bgSet(mirrorPath(uid, 'activeCourse'), activeCourse);
    // The active course changing (a different course became most-recently-
    // opened) means wrongAnswers needs to be re-watched against the new
    // course id — see watchWrongAnswersForCourse below.
    watchWrongAnswersForCourse(uid, activeCourse?.id || null);
  }, (err) => console.warn('homeMirror: activeCourse listener error:', err));
}

// ---- learnCourses/{activeCourseId}/wrongAnswers → homeData/{uid}/wrongCount ----
// Re-subscribed every time the active course id changes (including to
// null, when there's no active course), so this never keeps counting
// against a course that's no longer the one Home actually cares about.
function watchWrongAnswersForCourse(uid, courseId) {
  if (_wrongAnswersCourseId === courseId) return; // already watching the right course (or lack thereof)
  _wrongAnswersCourseId = courseId;
  if (_unsubWrongAnswers) { _unsubWrongAnswers(); _unsubWrongAnswers = null; }

  if (!courseId) {
    bgSet(mirrorPath(uid, 'wrongCount'), 0);
    return;
  }
  const q = collection(db, 'users', uid, 'learnCourses', courseId, 'wrongAnswers');
  _unsubWrongAnswers = onSnapshot(q, (snap) => {
    bgSet(mirrorPath(uid, 'wrongCount'), snap.size);
  }, (err) => console.warn('homeMirror: wrongAnswers listener error:', err));
}

// ---- friends → homeData/{uid}/friendReqCount + friendsCount ----
function watchFriends(uid) {
  const q = collection(db, 'users', uid, 'friends');
  return onSnapshot(q, (snap) => {
    let friendReqCount = 0;
    let friendsCount = 0;
    snap.forEach((d) => {
      const fd = d.data();
      if (fd.status === 'pending' && fd.direction === 'received') friendReqCount++;
      if (fd.status === 'accepted') friendsCount++;
    });
    bgUpdate(mirrorPath(uid), { friendReqCount, friendsCount });
  }, (err) => console.warn('homeMirror: friends listener error:', err));
}

// ---- sharedCourses (pending) → homeData/{uid}/sharedCourseCount ----
function watchSharedCourses(uid) {
  const q = query(collection(db, 'users', uid, 'sharedCourses'), where('status', '==', 'pending'));
  return onSnapshot(q, (snap) => {
    bgSet(mirrorPath(uid, 'sharedCourseCount'), snap.size);
  }, (err) => console.warn('homeMirror: sharedCourses listener error:', err));
}

// Call once right after sign-in (same lifecycle as initPremiumForCurrentUser,
// startXpListener, etc. in main.js's proceedAfterAuth()).
export function startHomeMirrorForCurrentUser() {
  stopHomeMirror();
  const u = auth.currentUser;
  if (!u) return;
  _unsubs = [
    watchLearnProfile(u.uid),
    watchActiveCourse(u.uid),
    watchFriends(u.uid),
    watchSharedCourses(u.uid),
  ];
}

// Call on sign-out / account switch (same lifecycle as resetPremiumState() etc.)
export function stopHomeMirror() {
  _unsubs.forEach((unsub) => unsub && unsub());
  _unsubs = [];
  if (_unsubWrongAnswers) { _unsubWrongAnswers(); _unsubWrongAnswers = null; }
  _wrongAnswersCourseId = null;
}