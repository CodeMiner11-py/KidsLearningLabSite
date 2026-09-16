// badges.js — Kids Learning Lab Badge System
//
// Self-contained module: owns its own Firestore reads/writes, injects its
// own <style> and DOM (profile shelf content, full badge-collection page,
// friends'-badges page, and the badge-unlock celebration screen), and is
// imported by main.js (account/friends/courses/profile UI) and learn.js
// (streaks/lessons/games). No circular imports — this file never imports
// from main.js or learn.js.
//
// ============================================================
// DATA MODEL
// ============================================================
// Private doc:  users/{uid}/badgesData/main
//   {
//     earned: { [badgeId]: { count, firstEarnedAt, lastEarnedAt } },
//     lessonsCompletedCount: number,
//     _backfilledLessons: bool,
//   }
// Public mirror (so friends can see, same pattern as xp/streak):
//   userProfiles/{uid}.badges = { [badgeId]: count }
//
// Every badge, once defined, has: id, name, description, icon
// (a Material Symbols icon name), category, and repeatable (bool).



import { db, auth } from './firebase.js';
import {
  doc, getDoc, setDoc, getDocs, collection, increment
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";

// ============================================================
// BADGE CATALOG — scalable, data-driven. Add new entries here to expand
// the system; nothing else needs to change for simple threshold badges.
// ============================================================
const CATEGORIES = {
  streak: { label: 'Streaks', icon: 'local_fire_department' },
  games: { label: 'Games', icon: 'sports_esports' },
  friends: { label: 'Friends', icon: 'group' },
  courses: { label: 'Courses', icon: 'auto_stories' },
  courseComplete: { label: 'Courses Completed', icon: 'workspace_premium' },
  lessons: { label: 'Lessons', icon: 'school' },
  unitComplete: { label: 'Units Completed', icon: 'task_alt' },
  account: { label: 'Account', icon: 'verified' },
};

// Mirrors learn.js's UNITS_PER_COURSE — badges.js is deliberately
// self-contained (no import from learn.js), so this is kept in sync by hand.
const UNITS_PER_COURSE = 10;

// Amount of XP awarded every time a NEW badge is unlocked (repeat unlocks
// of a repeatable badge count too, since each is still a fresh grant).
const BADGE_XP_REWARD = 100;

// ---- Streak badges: fixed early milestones, then every +100 days forever ----
const STREAK_MILESTONES = [5, 7, 20, 30, 90, 100, 150, 200];
function streakMilestonesUpTo(n) {
  const list = STREAK_MILESTONES.filter((m) => m <= n);
  for (let m = 300; m <= n; m += 100) list.push(m);
  return list;
}
function allKnownStreakMilestones(maxForDisplay) {
  // Used for rendering the "next locked streak badge" — only ever shows
  // milestones up to a bit past the user's current streak, since the
  // list is technically infinite.
  const list = [...STREAK_MILESTONES];
  for (let m = 300; m <= Math.max(maxForDisplay + 100, 300); m += 100) list.push(m);
  return list;
}
function streakBadgeName(days) {
  if (days === 30) return '1 Month Streak';
  if (days === 90) return '3 Month Streak';
  return `${days} Day Streak`;
}
function streakBadgeId(days) { return `streak_${days}`; }
// Streak badges of 100+ days additionally grant a name tag — a
// "[100+ DAYS]" / "[200+ DAYS]" / etc prefix shown next to the learner's
// name, same slot as the [PREMIUM ⭐️] badge (see formatDisplayNameWithTag
// below and PREMIUM_NAME_PREFIX in main.js). Below 100 days, badges are
// still earned exactly as before, just without a tag.
function hasNameTag(days) { return days >= 100; }
function nameTagLabel(days) { return `[${days}+ DAYS]`; }

// ---- Variable badge art (per the uploaded "7 DAYS" reference design) ----
// Every other badge category still just shows one shared Material Symbols
// glyph (a game controller, a book, etc — the icon genuinely doesn't vary
// per milestone there). Streak badges are different: each milestone gets
// its own little illustrated icon — a rounded-square badge with a flame
// glyph and the day count baked in underneath, like the uploaded "7 DAYS"
// reference. Built as inline SVG (no image assets needed) so it renders
// crisply at any size and themes with the rest of the app's CSS vars.
function streakBadgeArtSvg(days) {
  const label = days >= 100 ? `${days}+ DAYS` : `${days} DAY${days === 1 ? '' : 'S'}`;
  // Slightly bolder ring for the milestones that also carry a name tag, so
  // the "this one's special" tier reads at a glance in the collection
  // grid, not just in the description text.
  const isTagTier = hasNameTag(days);
  const ring = isTagTier ? '#F5B400' : 'var(--streak-orange, #FF8A2B)';
  return `
    <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" class="streak-badge-art">
      <rect x="4" y="4" width="92" height="92" rx="22" fill="var(--blue-pale, #EAF3FF)" stroke="${ring}" stroke-width="${isTagTier ? 4 : 2.5}"/>
      <path d="M50 20c7 9 11 16 11 23a11 11 0 11-22 0c0-2 .8-4 2-6 1 2 3 3 4 2-1-4 0-8 3-13.5z"
            fill="#FF6A1F" stroke="#FF4D00" stroke-width="1.5" stroke-linejoin="round"/>
      <path d="M46 32c3 4 4.5 7 4.5 9.5a4.5 4.5 0 11-9 0c0-1 .3-1.8.8-2.6.5 1 1.3 1.4 1.9 1-.4-1.8 0-4 1.8-7.9z"
            fill="#FFB01F"/>
      <text x="50" y="80" text-anchor="middle" font-size="15" font-weight="800" fill="#FFA028" font-family="var(--app-font, sans-serif)">${label}</text>
    </svg>
  `;
}

function streakBadgeDef(days) {
  return {
    id: streakBadgeId(days),
    name: streakBadgeName(days),
    description: hasNameTag(days)
      ? `Keep a learning streak going for ${days} days in a row. Earn a special name tag.`
      : `Keep a learning streak going for ${days} day${days === 1 ? '' : 's'} in a row.`,
    icon: 'local_fire_department',
    category: 'streak',
    repeatable: false,
    threshold: days,
  };
}

// Given the public { badgeId: count } summary mirror (userProfiles/{uid}.badges),
// returns the highest streak-nametag label this learner has earned, or null
// if they haven't hit 100 days yet. Exported so main.js can slot it into
// name rendering (own profile and friends' names alike) the same way it
// already slots in the Premium prefix.
export function highestStreakNameTag(badgesSummary) {
  if (!badgesSummary) return null;
  let best = 0;
  for (const key of Object.keys(badgesSummary)) {
    const m = /^streak_(\d+)$/.exec(key);
    if (!m) continue;
    const days = parseInt(m[1], 10);
    if (days > best && hasNameTag(days)) best = days;
  }
  return best > 0 ? nameTagLabel(best) : null;
}

// Same as highestStreakNameTag, but reads the signed-in user's own
// in-memory badgeData.earned (`{ [id]: { count, ... } }`) rather than a
// friend's public `{ id: count }` summary — for the one place (Settings /
// Change Name) that renders the current account's own tag without a
// Firestore round-trip.
export function myHighestStreakNameTag() {
  if (!badgeData?.earned) return null;
  const asSummary = {};
  for (const id of Object.keys(badgeData.earned)) asSummary[id] = 1;
  return highestStreakNameTag(asSummary);
}

// ---- Game badges — one entry per game. Add future games the same way. ----
const GAME_BADGE_DEFS = {
  maze: {
    id: 'game_first_maze', name: 'First Maze Game',
    description: 'Complete your first Maze game.', icon: 'route',
    category: 'games', repeatable: false,
  },
  trivia: {
    id: 'game_first_trivia', name: 'First Trivia Game',
    description: 'Complete your first Trivia game.', icon: 'quiz',
    category: 'games', repeatable: false,
  },
  voiceTrivia: {
    id: 'game_first_voice_trivia', name: 'First Voice Trivia Game',
    description: 'Complete your first Voice Trivia game.', icon: 'mic',
    category: 'games', repeatable: false,
  },
  connectors: {
    id: 'game_first_connectors', name: 'First Word Connectors Game',
    description: 'Complete your first Word Connectors game.', icon: 'hub',
    category: 'games', repeatable: false,
  },
  // Future games (Seesaw, Meltdown, Duel, etc.) plug in here with the same
  // shape — then call checkGameBadge('<key>') once from that game's
  // completion handler.
};

// ---- Friend badges ----
const FRIEND_MILESTONES = [1, 2, 3, 5, 10, 20];
const FRIEND_NAMES = { 1: 'First Friend', 2: 'Second Friend', 3: 'Third Friend', 5: 'Fifth Friend', 10: 'Tenth Friend', 20: 'Twentieth Friend' };
function friendBadgeDef(n) {
  return {
    id: `friend_${n}`, name: FRIEND_NAMES[n],
    description: `Add ${n} friend${n === 1 ? '' : 's'} on Kids Learning Lab.`,
    icon: 'group', category: 'friends', repeatable: false, threshold: n,
  };
}

// ---- Course-creation badges ----
const COURSE_MILESTONES = [1, 2, 3, 4, 5];
const COURSE_NAMES = { 1: 'First Course', 2: 'Second Course', 3: 'Third Course', 4: 'Fourth Course', 5: 'Fifth Course' };
function courseBadgeDef(n) {
  return {
    id: `course_${n}`, name: COURSE_NAMES[n],
    description: `Create ${n} course${n === 1 ? '' : 's'}.`,
    icon: 'auto_stories', category: 'courses', repeatable: false, threshold: n,
  };
}

// ---- Course-COMPLETION badges (finishing all units + the final course
// review) — distinct from courseBadgeDef() above, which is about how many
// courses a learner has CREATED. ----
const COURSE_COMPLETE_MILESTONES = [1, 2, 3, 4, 5, 10, 15, 20];
const COURSE_COMPLETE_NAMES = {
  1: 'First Course Complete!', 2: 'Second Course Complete!', 3: 'Third Course Complete!',
  4: 'Fourth Course Complete!', 5: 'Fifth Course Complete!', 10: '10th Course Complete!',
  15: '15th Course Complete!', 20: '20th Course Complete!',
};
function courseCompleteBadgeDef(n) {
  return {
    id: `course_complete_${n}`, name: COURSE_COMPLETE_NAMES[n],
    description: `Finish ${n} course${n === 1 ? '' : 's'} — every unit, lesson, and the final review.`,
    icon: 'workspace_premium', category: 'courseComplete', repeatable: false, threshold: n,
  };
}

// ---- Unit-completion badges (finishing every lesson in a unit plus its
// unit review) ----
const UNIT_COMPLETE_MILESTONES = [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 75, 100, 200, 300];
const UNIT_COMPLETE_NAMES = {
  1: 'First Unit Complete', 2: 'Second Unit Complete', 3: 'Third Unit Complete',
  4: 'Fourth Unit Complete', 5: 'Fifth Unit Complete', 10: '10th Unit Complete',
  15: '15th Unit Complete', 20: '20th Unit Complete', 25: '25th Unit Complete',
  50: '50th Unit Complete', 75: '75th Unit Complete', 100: '100th Unit Complete',
  200: '200th Unit Complete', 300: '300th Unit Complete',
};
function unitCompleteBadgeDef(n) {
  return {
    id: `unit_complete_${n}`, name: UNIT_COMPLETE_NAMES[n],
    description: `Finish ${n} unit${n === 1 ? '' : 's'} — every lesson plus the unit review.`,
    icon: 'task_alt', category: 'unitComplete', repeatable: false, threshold: n,
  };
}

// ---- Lesson-completion badges ----
const LESSON_MILESTONES = [1, 2, 3, 10, 20, 50, 100, 200, 300, 400, 500];
const LESSON_NAMES = {
  1: 'First Lesson', 2: 'Second Lesson', 3: 'Third Lesson', 10: 'Tenth Lesson', 20: 'Twentieth Lesson',
  50: '50th Lesson', 100: '100th Lesson', 200: '200th Lesson', 300: '300th Lesson', 400: '400th Lesson', 500: '500th Lesson',
};
function lessonBadgeDef(n) {
  return {
    id: `lesson_${n}`, name: LESSON_NAMES[n],
    description: `Complete ${n} lesson${n === 1 ? '' : 's'}.`,
    icon: 'school', category: 'lessons', repeatable: false, threshold: n,
  };
}

// ---- Account badge ----
const ACCOUNT_BADGE = {
  id: 'account_created', name: 'Account Created',
  description: 'Go to Learn to create your first learning course, or go to Listen to listen to Kids Learning Lab!',
  icon: 'verified', category: 'account', repeatable: false,
};

// ============================================================
// STATE
// ============================================================
let currentUid = null;
let badgeData = null; // { earned: {}, lessonsCompletedCount: 0 }
let pendingCelebrations = [];
let celebrationShowing = false;
// While paused, badges still get awarded/persisted/queued as normal — they
// just don't pop the celebration overlay yet. Lets learn.js finish showing
// the lesson-complete XP + streak overlays first, then resume so any badge
// celebrations appear as a tertiary step afterward instead of jumping on
// top of (and blocking) those overlays the instant a badge is earned.
let celebrationsPaused = false;
let onQueueDrainedCallback = null;

// Cached last-known stats, updated whenever a check*Badges function is
// called, purely so the collection page can render progress bars without
// needing learn.js/main.js to hand us their live state directly.
let lastStats = { streak: 0, lessons: 0, friends: 0, courses: 0, unitsCompleted: 0, coursesCompleted: 0, gamesDone: {} };

function escapeHtml(str) {
  return (str || '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ============================================================
// FIRESTORE
// ============================================================
function badgeDocRef(u) { return doc(db, 'users', u, 'badgesData', 'main'); }

export async function initBadgesForUser(uid) {
  if (!uid) return;
  currentUid = uid;
  const ref = badgeDocRef(uid);
  const snap = await getDoc(ref);
  if (snap.exists()) {
    badgeData = snap.data();
    if (!badgeData.earned || typeof badgeData.earned !== 'object') badgeData.earned = {};
    if (typeof badgeData.lessonsCompletedCount !== 'number') badgeData.lessonsCompletedCount = 0;
    if (typeof badgeData.unitsCompletedCount !== 'number') badgeData.unitsCompletedCount = 0;
    if (typeof badgeData.coursesCompletedCount !== 'number') badgeData.coursesCompletedCount = 0;
  } else {
    badgeData = { earned: {}, lessonsCompletedCount: 0, unitsCompletedCount: 0, coursesCompletedCount: 0 };
    await setDoc(ref, badgeData);
  }
  lastStats.lessons = badgeData.lessonsCompletedCount || 0;
  lastStats.unitsCompleted = badgeData.unitsCompletedCount || 0;
  lastStats.coursesCompleted = badgeData.coursesCompletedCount || 0;
  renderProfileShelf();
  await backfillProgressCountsIfNeeded();
}

// Called from main.js whenever the signed-in account changes, mirroring
// resetLearnState()'s job for this module's in-memory state.
export function resetBadgesState() {
  currentUid = null;
  badgeData = null;
  pendingCelebrations = [];
  celebrationShowing = false;
  celebrationsPaused = false;
  onQueueDrainedCallback = null;
  lastStats = { streak: 0, lessons: 0, friends: 0, courses: 0, unitsCompleted: 0, coursesCompleted: 0, gamesDone: {} };
  renderProfileShelf();
}

async function persist() {
  if (!currentUid || !badgeData) return;
  await setDoc(badgeDocRef(currentUid), badgeData, { merge: true }).catch((err) => console.error('Badge save failed:', err));
  const summary = {};
  for (const [id, v] of Object.entries(badgeData.earned)) summary[id] = v.count;
  await setDoc(doc(db, 'userProfiles', currentUid), { badges: summary }, { merge: true }).catch(() => {});
}

// Existing accounts predate per-lesson-completion (and unit/course
// completion) counting — this walks their existing course data once per
// counter (each independently flagged so it never re-runs) so badges
// retroactively reflect real progress instead of starting at 0.
async function backfillProgressCountsIfNeeded() {
  if (!badgeData || !currentUid) return;
  const needsLessons = !badgeData._backfilledLessons;
  const needsUnitsCourses = !badgeData._backfilledUnitsCourses;
  if (!needsLessons && !needsUnitsCourses) return;

  try {
    const coursesSnap = await getDocs(collection(db, 'users', currentUid, 'learnCourses'));
    let lessonCount = 0;
    let unitsCompleted = 0;
    let coursesCompleted = 0;

    for (const c of coursesSnap.docs) {
      const data = c.data() || {};
      if (needsUnitsCourses) {
        const cui = typeof data.currentUnitIndex === 'number' ? data.currentUnitIndex : 0;
        unitsCompleted += Math.min(cui, UNITS_PER_COURSE);
        if (data.courseReviewCompleted) coursesCompleted += 1;
      }
      if (needsLessons) {
        const lessonsSnap = await getDocs(collection(db, 'users', currentUid, 'learnCourses', c.id, 'lessons'));
        lessonsSnap.forEach((l) => { if (l.data()?.status === 'completed') lessonCount++; });
      }
    }

    if (needsLessons) {
      badgeData.lessonsCompletedCount = Math.max(badgeData.lessonsCompletedCount || 0, lessonCount);
      badgeData._backfilledLessons = true;
      lastStats.lessons = badgeData.lessonsCompletedCount;
    }
    if (needsUnitsCourses) {
      badgeData.unitsCompletedCount = Math.max(badgeData.unitsCompletedCount || 0, unitsCompleted);
      badgeData.coursesCompletedCount = Math.max(badgeData.coursesCompletedCount || 0, coursesCompleted);
      badgeData._backfilledUnitsCourses = true;
      lastStats.unitsCompleted = badgeData.unitsCompletedCount;
      lastStats.coursesCompleted = badgeData.coursesCompletedCount;
    }

    await persist();
    if (needsLessons) await awardAllLessonMilestonesUpTo(badgeData.lessonsCompletedCount);
    if (needsUnitsCourses) {
      await awardAllUnitCompleteMilestonesUpTo(badgeData.unitsCompletedCount);
      await awardAllCourseCompleteMilestonesUpTo(badgeData.coursesCompletedCount);
    }
  } catch (err) {
    console.error('Badge progress backfill failed:', err);
  }
}

function isEarned(id) { return !!(badgeData?.earned || {})[id]; }

async function award(def) {
  if (!badgeData) return null;
  const now = Date.now();
  const existing = badgeData.earned[def.id];
  if (existing) {
    if (!def.repeatable) return null;
    existing.count += 1;
    existing.lastEarnedAt = now;
  } else {
    badgeData.earned[def.id] = { count: 1, firstEarnedAt: now, lastEarnedAt: now };
  }
  await persist();
  await awardBadgeXp();
  renderProfileShelf();
  queueCelebration(def);
  return def;
}

// Atomic increment on the same learnProfile doc learn.js's XP awards use,
// so it can never race/clobber with a game or lesson XP write happening
// around the same time (learn.js's own XP writes are atomic increments
// too — see awardGameXp in learn.js).
async function awardBadgeXp() {
  if (!currentUid) return;
  const profileRef = doc(db, 'users', currentUid, 'learnProfile', 'main');
  await setDoc(profileRef, { xp: increment(BADGE_XP_REWARD) }, { merge: true })
    .catch((err) => console.error('Badge XP award failed:', err));
  await setDoc(doc(db, 'userProfiles', currentUid), { xp: increment(BADGE_XP_REWARD) }, { merge: true }).catch(() => {});
}

// ============================================================
// PUBLIC CHECK FUNCTIONS — call these from learn.js / main.js right after
// the relevant activity completes.
// ============================================================
export async function checkStreakBadges(streakCount) {
  if (!badgeData) return;
  lastStats.streak = streakCount || 0;
  for (const n of streakMilestonesUpTo(streakCount || 0)) {
    const def = streakBadgeDef(n);
    if (!isEarned(def.id)) await award(def);
  }
}

async function awardAllLessonMilestonesUpTo(count) {
  for (const n of LESSON_MILESTONES) {
    if (n <= count) {
      const def = lessonBadgeDef(n);
      if (!isEarned(def.id)) await award(def);
    }
  }
}

async function awardAllUnitCompleteMilestonesUpTo(count) {
  for (const n of UNIT_COMPLETE_MILESTONES) {
    if (n <= count) {
      const def = unitCompleteBadgeDef(n);
      if (!isEarned(def.id)) await award(def);
    }
  }
}

async function awardAllCourseCompleteMilestonesUpTo(count) {
  for (const n of COURSE_COMPLETE_MILESTONES) {
    if (n <= count) {
      const def = courseCompleteBadgeDef(n);
      if (!isEarned(def.id)) await award(def);
    }
  }
}

// Call once per real (non-review) lesson completion.
export async function checkLessonBadges() {
  if (!badgeData) return;
  badgeData.lessonsCompletedCount = (badgeData.lessonsCompletedCount || 0) + 1;
  lastStats.lessons = badgeData.lessonsCompletedCount;
  const n = badgeData.lessonsCompletedCount;
  if (LESSON_MILESTONES.includes(n)) {
    await award(lessonBadgeDef(n));
  } else {
    await persist();
  }
}

// Call once per unit finished (all its lessons plus its unit review).
export async function checkUnitCompletionBadges() {
  if (!badgeData) return;
  badgeData.unitsCompletedCount = (badgeData.unitsCompletedCount || 0) + 1;
  lastStats.unitsCompleted = badgeData.unitsCompletedCount;
  const n = badgeData.unitsCompletedCount;
  if (UNIT_COMPLETE_MILESTONES.includes(n)) {
    await award(unitCompleteBadgeDef(n));
  } else {
    await persist();
  }
}

// Call once per course finished (all 10 units plus the final course review).
export async function checkCourseCompletionBadges() {
  if (!badgeData) return;
  badgeData.coursesCompletedCount = (badgeData.coursesCompletedCount || 0) + 1;
  lastStats.coursesCompleted = badgeData.coursesCompletedCount;
  const n = badgeData.coursesCompletedCount;
  if (COURSE_COMPLETE_MILESTONES.includes(n)) {
    await award(courseCompleteBadgeDef(n));
  } else {
    await persist();
  }
}

// gameKey must match a key in GAME_BADGE_DEFS (e.g. 'maze', 'trivia', 'voiceTrivia').
export async function checkGameBadge(gameKey) {
  if (!badgeData) return;
  lastStats.gamesDone[gameKey] = true;
  const def = GAME_BADGE_DEFS[gameKey];
  if (!def) return;
  if (!isEarned(def.id)) await award(def);
}

export async function checkFriendBadges(acceptedCount) {
  if (!badgeData) return;
  lastStats.friends = acceptedCount || 0;
  for (const n of FRIEND_MILESTONES) {
    if (n <= acceptedCount) {
      const def = friendBadgeDef(n);
      if (!isEarned(def.id)) await award(def);
    }
  }
}

export async function checkCourseBadges(courseCount) {
  if (!badgeData) return;
  lastStats.courses = courseCount || 0;
  for (const n of COURSE_MILESTONES) {
    if (n <= courseCount) {
      const def = courseBadgeDef(n);
      if (!isEarned(def.id)) await award(def);
    }
  }
}

// onboardingComplete should be true once a user has both verified their
// email and chosen a username (main.js only lets a new signup reach the
// app after both — usernameSet on their userProfiles doc is proof of both).
export async function checkAccountBadge(onboardingComplete) {
  if (!badgeData || !onboardingComplete) return;
  if (!isEarned(ACCOUNT_BADGE.id)) await award(ACCOUNT_BADGE);
}

// ============================================================
// STYLE (injected once)
// ============================================================
function injectStylesOnce() {
  if (document.getElementById('badgesStyleTag')) return;
  const style = document.createElement('style');
  style.id = 'badgesStyleTag';
  style.textContent = `
    .badge-shelf-btn {
      width: 100%; display: block; background: none; border: none; cursor: pointer;
      font-family: 'Google Sans', sans-serif; text-align: left; padding: 0;
    }
    .badge-shelf-row {
      display: flex; gap: 12px; overflow-x: auto; padding: 16px 18px 4px;
      scrollbar-width: none;
    }
    .badge-shelf-row::-webkit-scrollbar { display: none; }
    .badge-shelf-empty { font-size: 12.5px; color: var(--ink-soft); padding: 4px 0 12px; }
    .badge-shelf-icon {
      flex-shrink: 0; width: 52px; height: 52px; border-radius: 16px;
      display: flex; align-items: center; justify-content: center;
      background: var(--blue-pale); color: var(--blue-main);
      font-size: 24px; position: relative;
    }
    .badge-shelf-icon .material-symbols-outlined { font-size: 24px; }
    .streak-badge-art { width: 100%; height: 100%; display: block; }
    .badge-shelf-icon-art, .badge-card-icon:has(.streak-badge-art) { padding: 0; overflow: hidden; }
    .badge-shelf-more {
      flex-shrink: 0; width: 52px; height: 52px; border-radius: 16px;
      display: flex; align-items: center; justify-content: center;
      background: #EEF4FC; color: var(--ink-soft); font-weight: 800; font-size: 12.5px;
    }
    .badge-shelf-footer {
      display: flex; align-items: center; justify-content: space-between;
      padding: 12px 18px 16px; border-top: 1.5px solid #EEF4FC; margin-top: 4px;
      font-weight: 700; font-size: 13.5px; color: var(--ink);
    }
    .badge-shelf-footer .material-symbols-outlined { font-size: 19px; color: var(--ink-soft); }

    .badges-page-sub { font-size: 13.5px; color: var(--ink-soft); margin: 0 0 14px; }
    .badges-summary-row {
      display: flex; align-items: center; justify-content: space-between;
      background: var(--white); border: 1.5px solid #DCE7F5; border-radius: 16px;
      padding: 14px 16px; margin-bottom: 16px;
    }
    .badges-summary-count { font-weight: 800; font-size: 15px; color: var(--blue-deep); }
    .badges-view-friends-btn {
      display: flex; align-items: center; gap: 6px;
      background: var(--blue-pale); color: var(--blue-main); border: none;
      border-radius: 12px; padding: 9px 14px; font-weight: 800; font-size: 12.5px;
      cursor: pointer; font-family: 'Google Sans', sans-serif;
    }
    .badges-category-title {
      font-weight: 800; font-size: 12px; color: var(--blue-main);
      text-transform: uppercase; letter-spacing: .06em; margin: 20px 0 10px;
    }
    .badges-grid {
      display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
    }
    .badge-card {
      background: var(--white); border: 1.5px solid #DCE7F5; border-radius: 16px;
      padding: 14px; display: flex; flex-direction: column; gap: 6px;
    }
    .badge-card.locked { opacity: 0.55; }
    .badge-card-icon {
      width: 40px; height: 40px; border-radius: 12px;
      display: flex; align-items: center; justify-content: center;
      background: var(--blue-pale); color: var(--blue-main); font-size: 21px;
    }
    .badge-card.locked .badge-card-icon { background: #EEF4FC; color: var(--ink-soft); }
    .badge-card-name { font-weight: 800; font-size: 13px; color: var(--ink); }
    .badge-card-desc { font-size: 11.5px; color: var(--ink-soft); line-height: 1.35; }
    .badge-card-meta { font-size: 11px; color: var(--blue-main); font-weight: 700; margin-top: 2px; }
    .badge-card-progress-bar {
      width: 100%; height: 6px; border-radius: 4px; background: #EEF4FC; overflow: hidden; margin-top: 2px;
    }
    .badge-card-progress-fill { height: 100%; background: var(--blue-main); border-radius: 4px; }

    /* Badge-unlock celebration is a full-screen overlay (same
       kll-modal-overlay/lesson-summary-overlay pattern used by the
       Lesson Complete and Streak Extended screens elsewhere in the app) —
       not a floating centered dialog. These rules only style the
       badge-specific pieces dropped inside that shared layout. */
    /* It can be triggered while the Lesson Complete / Streak Extended
       overlays are still in the DOM (mid-sequence), so it needs to sit
       above those, not just above whatever was showing when it was
       first created. */
    #badgeCelebOverlay { z-index: 9600; }
    .badge-celeb-icon {
      width: 88px; height: 88px; border-radius: 50%; margin: 0 auto 18px;
      background: linear-gradient(135deg, var(--blue-bright), var(--blue-main));
      display: flex; align-items: center; justify-content: center;
      color: var(--white); font-size: 42px;
    }
    .badge-celeb-eyebrow {
      font-weight: 800; font-size: 12px; color: var(--blue-main);
      text-transform: uppercase; letter-spacing: .06em; margin-bottom: 6px;
      text-align: center;
    }
    .badge-celeb-name { font-size: 20px; font-weight: 800; color: var(--blue-deep); margin: 0 0 8px; text-align: center; }
    .badge-celeb-desc { font-size: 13.5px; color: var(--ink-soft); line-height: 1.4; margin-bottom: 24px; text-align: center; }
    .badge-celeb-xp {
      display: inline-flex; align-items: center; gap: 5px;
      background: var(--blue-pale); color: var(--blue-main);
      border-radius: 999px; padding: 6px 14px; font-weight: 800; font-size: 13px;
      margin: -8px 0 4px;
    }
    .badge-celeb-xp .material-symbols-outlined { font-size: 16px; }

    .friend-badge-row {
      display: flex; align-items: center; gap: 12px; padding: 14px 4px;
      border-bottom: 1.5px solid #EEF4FC; cursor: pointer; background: none; border-left: none; border-right: none; border-top: none;
      width: 100%; text-align: left; font-family: 'Google Sans', sans-serif;
    }
    .friend-badge-icon {
      width: 40px; height: 40px; border-radius: 50%; background: var(--blue-pale); color: var(--blue-main);
      display: flex; align-items: center; justify-content: center; font-size: 19px; flex-shrink: 0;
    }
    .friend-badge-name { font-weight: 800; font-size: 14px; color: var(--ink); }
    .friend-badge-sub { font-size: 12px; color: var(--ink-soft); margin-top: 1px; }
  `;
  document.head.appendChild(style);
}

// ============================================================
// CELEBRATION SCREEN — a full-page overlay layered on top of whatever
// Lesson/Game Complete screen is already showing.
// ============================================================
let celebEls = null;
function ensureCelebrationDom() {
  if (celebEls) return celebEls;
  injectStylesOnce();
  // Reuses the same full-screen "lesson summary" overlay shell (defined
  // globally in index.html's stylesheet, already used by the Lesson
  // Complete and Streak Extended screens) instead of a centered floating
  // card, so a badge unlock reads as its own celebration page rather than
  // a modal popup.
  const overlay = document.createElement('div');
  overlay.className = 'kll-modal-overlay lesson-summary-overlay';
  overlay.id = 'badgeCelebOverlay';
  overlay.innerHTML = `
    <div class="lesson-summary-container" id="badgeCelebContainer">
      <div class="lesson-summary-body">
        <div class="badge-celeb-icon"><span class="material-symbols-outlined" id="badgeCelebIcon">military_tech</span></div>
        <div class="badge-celeb-eyebrow">You earned a new badge!</div>
        <h2 class="badge-celeb-name" id="badgeCelebName"></h2>
        <p class="badge-celeb-desc" id="badgeCelebDesc"></p>
        <div class="badge-celeb-xp"><span class="material-symbols-outlined">bolt</span> +${BADGE_XP_REWARD} XP</div>
      </div>
      <div class="lesson-summary-footer">
        <button class="lesson-action-btn" id="badgeCelebContinueBtn">Continue</button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);
  celebEls = {
    overlay,
    icon: overlay.querySelector('#badgeCelebIcon'),
    name: overlay.querySelector('#badgeCelebName'),
    desc: overlay.querySelector('#badgeCelebDesc'),
    continueBtn: overlay.querySelector('#badgeCelebContinueBtn'),
  };
  return celebEls;
}

function queueCelebration(def) {
  pendingCelebrations.push(def);
  processCelebrationQueue();
}

// Call to hold off showing any badge celebration overlay until
// resumeCelebrations() is called — badges still get earned/persisted in the
// meantime, they just wait to display.
export function pauseCelebrations() {
  celebrationsPaused = true;
}

// Unpauses and starts showing whatever celebrations piled up while paused.
// If onAllDone is provided, it fires once the queue is fully drained (the
// last celebration's Continue is tapped) — or immediately if there was
// nothing queued at all.
export function resumeCelebrations(onAllDone) {
  celebrationsPaused = false;
  if (onAllDone) {
    if (celebrationShowing || pendingCelebrations.length) {
      onQueueDrainedCallback = onAllDone;
    } else {
      onAllDone();
      return;
    }
  }
  processCelebrationQueue();
}

function processCelebrationQueue() {
  if (celebrationsPaused || celebrationShowing) return;
  const def = pendingCelebrations.shift();
  if (!def) {
    if (onQueueDrainedCallback) {
      const cb = onQueueDrainedCallback;
      onQueueDrainedCallback = null;
      cb();
    }
    return;
  }
  celebrationShowing = true;
  const els = ensureCelebrationDom();
  els.icon.innerHTML = (def.category === 'streak' && def.threshold) ? streakBadgeArtSvg(def.threshold) : `<span class="material-symbols-outlined">${def.icon || 'military_tech'}</span>`;
  els.name.textContent = def.name;
  els.desc.textContent = def.description;
  els.overlay.classList.add('show');
  els.continueBtn.onclick = () => {
    els.overlay.classList.remove('show');
    celebrationShowing = false;
    processCelebrationQueue();
  };
}

// ============================================================
// PROFILE SHELF (mounted into the existing #profileBadgesShelf in
// index.html's Profile page, above the About card)
// ============================================================
function renderProfileShelf() {
  const shelf = document.getElementById('profileBadgesShelf');
  const countLabel = document.getElementById('profileBadgesCountLabel');
  if (!shelf) return;
  injectStylesOnce();

  const earnedIds = badgeData ? Object.keys(badgeData.earned || {}) : [];
  if (!earnedIds.length) {
    shelf.innerHTML = `<div class="badge-shelf-empty">Complete lessons, games, and more to start earning badges!</div>`;
  } else {
    const shown = earnedIds.slice(0, 8);
    shelf.innerHTML = shown.map((id) => {
      const def = allBadgeDefsById()[id];
      if (def?.category === 'streak' && def.threshold) {
        return `<div class="badge-shelf-icon badge-shelf-icon-art">${streakBadgeArtSvg(def.threshold)}</div>`;
      }
      const icon = def?.icon || 'military_tech';
      return `<div class="badge-shelf-icon"><span class="material-symbols-outlined">${icon}</span></div>`;
    }).join('') + (earnedIds.length > shown.length ? `<div class="badge-shelf-more">+${earnedIds.length - shown.length}</div>` : '');
  }
  if (countLabel) {
    countLabel.textContent = `${earnedIds.length} badge${earnedIds.length === 1 ? '' : 's'} earned`;
  }
}

const profileBadgesBtnEl = document.getElementById('profileBadgesBtn');
profileBadgesBtnEl?.addEventListener('click', () => openBadgesCollectionPage());

// ============================================================
// FULL BADGE COLLECTION PAGE (own badges)
// ============================================================
// Every badge definition currently relevant to display: earned ones (any
// category) plus the *next* locked one per category/track, so the page
// stays a manageable length even though streak/lesson milestones are long.
function allBadgeDefsById() {
  const map = {};
  for (const n of allKnownStreakMilestones(lastStats.streak)) { const d = streakBadgeDef(n); map[d.id] = d; }
  for (const key of Object.keys(GAME_BADGE_DEFS)) { const d = GAME_BADGE_DEFS[key]; map[d.id] = { ...d, gameKey: key }; }
  for (const n of FRIEND_MILESTONES) { const d = friendBadgeDef(n); map[d.id] = d; }
  for (const n of COURSE_MILESTONES) { const d = courseBadgeDef(n); map[d.id] = d; }
  for (const n of COURSE_COMPLETE_MILESTONES) { const d = courseCompleteBadgeDef(n); map[d.id] = d; }
  for (const n of LESSON_MILESTONES) { const d = lessonBadgeDef(n); map[d.id] = d; }
  for (const n of UNIT_COMPLETE_MILESTONES) { const d = unitCompleteBadgeDef(n); map[d.id] = d; }
  map[ACCOUNT_BADGE.id] = ACCOUNT_BADGE;
  return map;
}

function badgeCardHtml(def) {
  const earnedInfo = badgeData?.earned?.[def.id];
  const earned = !!earnedInfo;
  let metaHtml = '';
  if (earned) {
    const dateStr = new Date(earnedInfo.firstEarnedAt).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
    metaHtml = `<div class="badge-card-meta">Earned ${escapeHtml(dateStr)}${earnedInfo.count > 1 ? ` · ×${earnedInfo.count}` : ''}</div>`;
  } else {
    metaHtml = progressHtml(def);
  }
  return `
    <div class="badge-card ${earned ? '' : 'locked'}">
      <div class="badge-card-icon">${(def.category === 'streak' && def.threshold) ? streakBadgeArtSvg(def.threshold) : `<span class="material-symbols-outlined">${def.icon}</span>`}</div>
      <div class="badge-card-name">${escapeHtml(def.name)}</div>
      <div class="badge-card-desc">${escapeHtml(def.description)}</div>
      ${metaHtml}
    </div>
  `;
}

function progressHtml(def) {
  let current = 0, threshold = def.threshold;
  if (def.category === 'streak') current = lastStats.streak;
  else if (def.category === 'lessons') current = lastStats.lessons;
  else if (def.category === 'friends') current = lastStats.friends;
  else if (def.category === 'courses') current = lastStats.courses;
  else if (def.category === 'courseComplete') current = lastStats.coursesCompleted;
  else if (def.category === 'unitComplete') current = lastStats.unitsCompleted;
  else if (def.category === 'games') {
    return `<div class="badge-card-meta">${lastStats.gamesDone[def.gameKey] ? 'Almost there!' : 'Not yet completed'}</div>`;
  } else if (def.category === 'account') {
    return `<div class="badge-card-meta">Complete onboarding to unlock</div>`;
  }
  if (!threshold) return '';
  const pct = Math.min(100, Math.round((current / threshold) * 100));
  return `
    <div class="badge-card-meta">${current}/${threshold}</div>
    <div class="badge-card-progress-bar"><div class="badge-card-progress-fill" style="width:${pct}%"></div></div>
  `;
}

function nextLockedByCategory(category) {
  // Returns the single next not-yet-earned def for a track, so the page
  // shows "what's coming up" rather than every future milestone at once.
  let candidates = [];
  if (category === 'streak') candidates = allKnownStreakMilestones(lastStats.streak).map(streakBadgeDef);
  else if (category === 'friends') candidates = FRIEND_MILESTONES.map(friendBadgeDef);
  else if (category === 'courses') candidates = COURSE_MILESTONES.map(courseBadgeDef);
  else if (category === 'courseComplete') candidates = COURSE_COMPLETE_MILESTONES.map(courseCompleteBadgeDef);
  else if (category === 'lessons') candidates = LESSON_MILESTONES.map(lessonBadgeDef);
  else if (category === 'unitComplete') candidates = UNIT_COMPLETE_MILESTONES.map(unitCompleteBadgeDef);
  for (const def of candidates) {
    if (!isEarned(def.id)) return def;
  }
  return null;
}

function renderBadgesCollectionBody() {
  const earnedIds = badgeData ? Object.keys(badgeData.earned || {}) : [];
  const allDefs = allBadgeDefsById();
  const totalKnown = Object.keys(allDefs).length;

  let html = `
    <div class="badges-summary-row">
      <div class="badges-summary-count">${earnedIds.length} badge${earnedIds.length === 1 ? '' : 's'} earned</div>
      <button type="button" class="badges-view-friends-btn" id="badgesViewFriendsBtn">
        <span class="material-symbols-outlined" style="font-size:16px;">group</span> View Friends
      </button>
    </div>
    <p class="badges-page-sub">Earn badges by building streaks, finishing lessons and units, playing games, adding friends, and creating and completing courses. Every badge is worth +${BADGE_XP_REWARD} XP.</p>
  `;

  for (const catKey of Object.keys(CATEGORIES)) {
    const cat = CATEGORIES[catKey];
    const earnedInCat = Object.values(allDefs).filter((d) => d.category === catKey && isEarned(d.id));
    let lockedInCat = [];
    if (catKey === 'games') {
      lockedInCat = Object.entries(GAME_BADGE_DEFS)
        .filter(([, d]) => !isEarned(d.id))
        .map(([gameKey, d]) => ({ ...d, gameKey }));
    } else if (catKey === 'account') {
      lockedInCat = isEarned(ACCOUNT_BADGE.id) ? [] : [ACCOUNT_BADGE];
    } else {
      const next = nextLockedByCategory(catKey);
      lockedInCat = next ? [next] : [];
    }
    const cards = [...earnedInCat, ...lockedInCat];
    if (!cards.length) continue;
    html += `<div class="badges-category-title">${escapeHtml(cat.label)}</div><div class="badges-grid">`;
    html += cards.map(badgeCardHtml).join('');
    html += `</div>`;
  }

  return html;
}

let badgesPageEls = null;
function ensureBadgesCollectionPageDom() {
  if (badgesPageEls) return badgesPageEls;
  injectStylesOnce();
  const overlay = document.createElement('div');
  overlay.className = 'kll-modal-overlay games-page-overlay';
  overlay.id = 'badgesCollectionPageOverlay';
  overlay.innerHTML = `
    <div class="games-page-container">
      <div class="games-page-topbar">
        <button id="badgesCollectionExitBtn" class="games-exit-btn">
          <span class="material-symbols-outlined">close</span>
        </button>
        <div class="games-page-title">Your Badges</div>
      </div>
      <div class="games-page-body" id="badgesCollectionBody"></div>
    </div>
  `;
  document.body.appendChild(overlay);
  overlay.querySelector('#badgesCollectionExitBtn').addEventListener('click', () => overlay.classList.remove('show'));
  badgesPageEls = { overlay, body: overlay.querySelector('#badgesCollectionBody') };
  return badgesPageEls;
}

export function openBadgesCollectionPage() {
  const els = ensureBadgesCollectionPageDom();
  els.body.innerHTML = renderBadgesCollectionBody();
  els.body.querySelector('#badgesViewFriendsBtn')?.addEventListener('click', openFriendsBadgesListPage);
  els.overlay.classList.add('show');
}

// ============================================================
// FRIENDS' BADGES — "View Friends" button on the collection page opens a
// list of accepted friends; tapping one shows their earned badges
// (read-only, sourced from the public userProfiles.badges mirror).
// ============================================================
let friendsBadgesListEls = null;
function ensureFriendsBadgesListDom() {
  if (friendsBadgesListEls) return friendsBadgesListEls;
  injectStylesOnce();
  const overlay = document.createElement('div');
  overlay.className = 'kll-modal-overlay games-page-overlay';
  overlay.id = 'friendsBadgesListOverlay';
  overlay.innerHTML = `
    <div class="games-page-container">
      <div class="games-page-topbar">
        <button id="friendsBadgesListExitBtn" class="games-exit-btn">
          <span class="material-symbols-outlined">close</span>
        </button>
        <div class="games-page-title">Friends' Badges</div>
      </div>
      <div class="games-page-body" id="friendsBadgesListBody"></div>
    </div>
  `;
  document.body.appendChild(overlay);
  overlay.querySelector('#friendsBadgesListExitBtn').addEventListener('click', () => overlay.classList.remove('show'));
  friendsBadgesListEls = { overlay, body: overlay.querySelector('#friendsBadgesListBody') };
  return friendsBadgesListEls;
}

async function openFriendsBadgesListPage() {
  const els = ensureFriendsBadgesListDom();
  els.body.innerHTML = `<p class="badges-page-sub">Loading friends…</p>`;
  els.overlay.classList.add('show');

  if (!currentUid) { els.body.innerHTML = `<p class="badges-page-sub">Sign in to see friends' badges.</p>`; return; }
  try {
    const snap = await getDocs(collection(db, 'users', currentUid, 'friends'));
    const accepted = [];
    snap.forEach((d) => { if (d.data()?.status === 'accepted') accepted.push(d.id); });
    if (!accepted.length) {
      els.body.innerHTML = `<p class="badges-page-sub">Add some friends to see their badge collections here.</p>`;
      return;
    }
    let rowsHtml = '';
    for (const otherUid of accepted) {
      const profSnap = await getDoc(doc(db, 'userProfiles', otherUid)).catch(() => null);
      const info = profSnap && profSnap.exists() ? profSnap.data() : {};
      const badgeCount = info.badges ? Object.keys(info.badges).length : 0;
      rowsHtml += `
        <button type="button" class="friend-badge-row" data-uid="${escapeHtml(otherUid)}">
          <div class="friend-badge-icon"><span class="material-symbols-outlined">emoji_events</span></div>
          <div>
            <div class="friend-badge-name">${escapeHtml(info.displayName || info.email || 'Learner')}</div>
            <div class="friend-badge-sub">${badgeCount} badge${badgeCount === 1 ? '' : 's'} earned</div>
          </div>
        </button>
      `;
    }
    els.body.innerHTML = rowsHtml;
    els.body.querySelectorAll('.friend-badge-row').forEach((row) => {
      row.addEventListener('click', () => openFriendBadgesDetailPage(row.dataset.uid));
    });
  } catch (err) {
    console.error('Failed to load friends for badges page:', err);
    els.body.innerHTML = `<p class="badges-page-sub">Could not load friends right now.</p>`;
  }
}

let friendBadgesDetailEls = null;
function ensureFriendBadgesDetailDom() {
  if (friendBadgesDetailEls) return friendBadgesDetailEls;
  injectStylesOnce();
  const overlay = document.createElement('div');
  overlay.className = 'kll-modal-overlay games-page-overlay';
  overlay.id = 'friendBadgesDetailOverlay';
  overlay.innerHTML = `
    <div class="games-page-container">
      <div class="games-page-topbar">
        <button id="friendBadgesDetailExitBtn" class="games-exit-btn">
          <span class="material-symbols-outlined">close</span>
        </button>
        <div class="games-page-title" id="friendBadgesDetailTitle">Friend's Badges</div>
      </div>
      <div class="games-page-body" id="friendBadgesDetailBody"></div>
    </div>
  `;
  document.body.appendChild(overlay);
  overlay.querySelector('#friendBadgesDetailExitBtn').addEventListener('click', () => overlay.classList.remove('show'));
  friendBadgesDetailEls = { overlay, title: overlay.querySelector('#friendBadgesDetailTitle'), body: overlay.querySelector('#friendBadgesDetailBody') };
  return friendBadgesDetailEls;
}

// Renders a badge-map ({ badgeId: count }) into the same badge-grid markup
// used everywhere else (own collection, friend detail page). Exported so
// other screens (e.g. the friend profile page in main.js) can drop a
// friend's badges inline without navigating through the badges page.
export function renderBadgeGridHtml(badgeMap) {
  injectStylesOnce(); // callers outside this module (e.g. friend profile page) won't have these classes yet
  const ids = Object.keys(badgeMap || {});
  if (!ids.length) return `<p class="badges-page-sub">No badges earned yet.</p>`;
  const allDefs = allBadgeDefsById();
  return `<div class="badges-grid">` + ids.map((id) => {
    const def = allDefs[id] || { name: 'Badge', description: '', icon: 'military_tech' };
    const count = badgeMap[id];
    return `
      <div class="badge-card">
        <div class="badge-card-icon">${(def.category === 'streak' && def.threshold) ? streakBadgeArtSvg(def.threshold) : `<span class="material-symbols-outlined">${def.icon}</span>`}</div>
        <div class="badge-card-name">${escapeHtml(def.name)}</div>
        <div class="badge-card-desc">${escapeHtml(def.description)}</div>
        ${count > 1 ? `<div class="badge-card-meta">×${count}</div>` : ''}
      </div>
    `;
  }).join('') + `</div>`;
}

async function openFriendBadgesDetailPage(otherUid) {
  const els = ensureFriendBadgesDetailDom();
  els.body.innerHTML = `<p class="badges-page-sub">Loading…</p>`;
  els.overlay.classList.add('show');
  try {
    const profSnap = await getDoc(doc(db, 'userProfiles', otherUid));
    const info = profSnap.exists() ? profSnap.data() : {};
    els.title.textContent = info.displayName ? `${info.displayName}'s Badges` : "Friend's Badges";
    els.body.innerHTML = renderBadgeGridHtml(info.badges || {});
  } catch (err) {
    console.error('Failed to load friend badge detail:', err);
    els.body.innerHTML = `<p class="badges-page-sub">Could not load this friend's badges.</p>`;
  }
}