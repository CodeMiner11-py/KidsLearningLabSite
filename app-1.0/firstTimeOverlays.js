// firstTimeOverlays.js — Kids Learning Lab "first time you do X" full-page
// overlays.
//
// Net-new module (no existing app code touches this). Clones the
// established kll-modal-overlay + lesson-summary-* full-pager pattern
// already used by Lesson Complete / Streak Extended / Badge Celebration
// (see badges.js) — NOT the smaller floating kll-modal card style, and NOT
// the paywall's own bespoke markup. One flag per overlay, stored in a
// single Firestore map so this scales to more overlays later without
// schema churn, and so pre-existing accounts (created before this system
// shipped) still see every one of these the first real time they hit that
// trigger — a missing/false flag always means "show it".
//
// ============================================================
// DATA MODEL
// ============================================================
//   users/{uid}/firstTimeOverlays/main
//     { seen: { [overlayId]: true } }
//
// ============================================================
// PUBLIC API
// ============================================================
//   initFirstTimeOverlaysForUser(uid)   — call once on sign-in, mirrors the
//                                          init pattern in badges.js/friends.js
//   resetFirstTimeOverlaysState()       — call on sign-out
//   maybeShowOverlay(id, opts)          — shows the overlay if unseen, marks
//                                          it seen, and resolves once
//                                          dismissed. Resolves immediately
//                                          (no-op) if already seen.
//   hasSeenOverlay(id)                  — sync check, for callers that need
//                                          to branch without awaiting
//
// opts:
//   courseColor  — CSS color string; overrides the accent for overlays that
//                   should theme to the active course (New Course Created)
//   vars         — { [token]: string } simple {{token}} substitutions for
//                   the body copy (e.g. days count, email, course name)

import { db } from './firebase.js';
import { doc, getDoc, setDoc } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";

// ============================================================
// STATE
// ============================================================
let currentUid = null;
let seenMap = {}; // { [overlayId]: true }

function docRef(u) { return doc(db, 'users', u, 'firstTimeOverlays', 'main'); }

export async function initFirstTimeOverlaysForUser(uid) {
  if (!uid) return;
  currentUid = uid;
  const snap = await getDoc(docRef(uid));
  seenMap = (snap.exists() && snap.data()?.seen) || {};
}

export function resetFirstTimeOverlaysState() {
  currentUid = null;
  seenMap = {};
}

export function hasSeenOverlay(id) {
  return !!seenMap[id];
}

async function markSeen(id) {
  seenMap[id] = true;
  if (!currentUid) return;
  await setDoc(docRef(currentUid), { seen: { [id]: true } }, { merge: true })
    .catch((err) => console.error('firstTimeOverlays: failed to persist seen flag', id, err));
}

// ============================================================
// ICONS — hand-drawn inline SVGs (24x24 viewBox, single accent color via
// currentColor, ~1.8-2px stroke to match Material Symbols Outlined's
// visual weight elsewhere in the app) so these overlays don't rely on
// emoji or on Material Symbols glyphs for their centerpiece art, per spec.
// ============================================================
const ICONS = {
  // Personalized Review — lightning bolt
  bolt: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M13 2L4 14h6l-1 8 9-12h-6l1-8z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" fill="currentColor" fill-opacity="0.15"/>
  </svg>`,
  // Streak Pass used yesterday — flame inside an ice cube
  iceFlame: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M4.5 6.5L12 3l7.5 3.5v9L12 21l-7.5-3.5v-9z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" fill="currentColor" fill-opacity="0.08"/>
    <path d="M12 7c1.4 1.7 2.2 3 2.2 4.3a2.2 2.2 0 11-4.4 0c0-.5.2-1 .5-1.5.3.5.7.7 1 .5-.2-.9 0-2 .7-3.3z" fill="currentColor" fill-opacity="0.55" stroke="currentColor" stroke-width="1.2" stroke-linejoin="round"/>
  </svg>`,
  // New Course Created — checkmark burst
  courseCheck: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="12" cy="12" r="8.5" stroke="currentColor" stroke-width="1.8" fill="currentColor" fill-opacity="0.12"/>
    <path d="M8.5 12.2l2.4 2.4 4.6-5.2" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>
  </svg>`,
  // Welcome Back — atom + flask (mirrors the app icon's science motif)
  atomFlask: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <ellipse cx="12" cy="12" rx="9" ry="3.6" stroke="currentColor" stroke-width="1.5"/>
    <ellipse cx="12" cy="12" rx="9" ry="3.6" stroke="currentColor" stroke-width="1.5" transform="rotate(60 12 12)"/>
    <ellipse cx="12" cy="12" rx="9" ry="3.6" stroke="currentColor" stroke-width="1.5" transform="rotate(120 12 12)"/>
    <circle cx="12" cy="12" r="2.1" fill="currentColor"/>
  </svg>`,
  // Premium welcome — ribbon/medal
  ribbon: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="12" cy="9" r="6" stroke="currentColor" stroke-width="1.8" fill="currentColor" fill-opacity="0.12"/>
    <path d="M12 5.5l1.1 2.3 2.5.4-1.8 1.8.4 2.5-2.2-1.2-2.2 1.2.4-2.5-1.8-1.8 2.5-.4L12 5.5z" fill="currentColor"/>
    <path d="M8.5 14.5L7 22l5-2.6 5 2.6-1.5-7.5" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>
  </svg>`,
  // Course Sent — share arrow out of a box
  shareOut: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 3v11" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    <path d="M8 7l4-4 4 4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M5 13v6a2 2 0 002 2h10a2 2 0 002-2v-6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>`,
  // QR Processing / Account Creation spinner — corner-bracket frame
  scanFrame: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M4 8V5a1 1 0 011-1h3M20 8V5a1 1 0 00-1-1h-3M4 16v3a1 1 0 001 1h3M20 16v3a1 1 0 01-1 1h-3" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    <rect x="8.5" y="8.5" width="3" height="3" rx="0.5" fill="currentColor"/>
    <rect x="13" y="8.5" width="2.5" height="3" rx="0.5" fill="currentColor" fill-opacity="0.5"/>
    <rect x="8.5" y="13" width="3" height="2.5" rx="0.5" fill="currentColor" fill-opacity="0.5"/>
  </svg>`,
  // Learning Games — joystick/controller
  controller: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M6 9.5h5.5M8.75 7v5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
    <circle cx="16" cy="8.7" r="1" fill="currentColor"/>
    <circle cx="18.3" cy="11" r="1" fill="currentColor"/>
    <path d="M4.5 9.2c0-2 1.4-3.2 3.4-3.2h8.2c2.6 0 4.9 2.1 5.3 4.7l.5 3.2c.3 2-1.2 3.6-3 3.1a3.3 3.3 0 01-1.6-1l-1-1a3 3 0 00-2.2-.9H9.9a3 3 0 00-2.2.9l-1 1a3.3 3.3 0 01-1.6 1c-1.8.5-3.3-1.1-3-3.1l.5-3.2c.2-1 .5-1.7.9-2.5z" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round" fill="currentColor" fill-opacity="0.08"/>
  </svg>`,
  // Streaks — flame
  flame: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 2.5c1.8 2.3 2.8 4 2.8 5.7 0 1-.5 1.7-1.1 2.3.9-.2 1.6-1 1.9-1.9 1.6 1.8 2.4 3.5 2.4 5.2 0 3.7-2.9 6.7-6.5 6.7S5 17.5 5 13.8c0-2.5 1-4.3 2.4-6 .2 1 .8 1.7 1.6 2 -.6-.9-.9-1.9-.9-2.9 0-1.6.9-3 1.9-4.4.6.9 1.3 1.7 2 2z" fill="currentColor" fill-opacity="0.18" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/>
  </svg>`,
  // Customize Your Avatar — palette
  palette: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 3.5c-4.7 0-8.5 3.6-8.5 8 0 3.3 2.3 4.7 4 4.7.9 0 1.2-.5 1.2-1.1 0-.5-.3-.9-.3-1.6 0-1.2 1-2 2.3-2h2.1c2.7 0 5.2-1.9 5.2-5C18 6.2 15.5 3.5 12 3.5z" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round" fill="currentColor" fill-opacity="0.1"/>
    <circle cx="8.3" cy="9" r="1.1" fill="currentColor"/>
    <circle cx="11.7" cy="7" r="1.1" fill="currentColor"/>
    <circle cx="15" cy="9" r="1.1" fill="currentColor"/>
    <circle cx="9" cy="13" r="0.9" fill="currentColor"/>
  </svg>`,
  // Add Friends — two people
  people: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="8.7" cy="8" r="2.7" stroke="currentColor" stroke-width="1.7" fill="currentColor" fill-opacity="0.12"/>
    <circle cx="16" cy="9.5" r="2.1" stroke="currentColor" stroke-width="1.6" fill="currentColor" fill-opacity="0.12"/>
    <path d="M3.5 19c.4-3 2.6-4.8 5.2-4.8s4.8 1.8 5.2 4.8" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
    <path d="M14.8 14.6c2 .2 3.5 1.7 3.9 4.4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
  </svg>`,
  // Earn Badges — star ribbon
  starRibbon: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M9 15L6.5 22l5.5-3 5.5 3L15 15" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round" fill="currentColor" fill-opacity="0.1"/>
    <path d="M12 2l2.1 4.3 4.7.7-3.4 3.3.8 4.7L12 12.7l-4.2 2.3.8-4.7-3.4-3.3 4.7-.7L12 2z" fill="currentColor" fill-opacity="0.2" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>
  </svg>`,
  // Adaptive Difficulty — dial/gauge
  dial: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M4 15a8 8 0 1116 0" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
    <path d="M12 15l4-5.2" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    <circle cx="12" cy="15" r="1.4" fill="currentColor"/>
    <path d="M4 15h1.6M18.4 15H20M6.3 9.3l1.1 1.1M17.7 9.3l-1.1 1.1" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
  </svg>`,
};

// ============================================================
// OVERLAY CATALOG — data-driven, matches the PDF spec 1:1. `trigger`
// documents where each is fired from (for the wiring pass); not read by
// this module.
// ============================================================
const OVERLAY_DEFS = {
  personalizedReviewStart: {
    icon: 'bolt', color: 'var(--purple)',
    title: 'Personalized Review\nStarting…',
    body: '',
    buttonLabel: null, // no button — this one auto-advances, see AUTO_ADVANCE below
  },
  streakPassUsedYesterday: {
    icon: 'iceFlame', color: 'var(--blue-main)',
    title: 'Used a Streak Pass yesterday…',
    body: 'Try not to use one today!',
    buttonLabel: 'Continue',
  },
  newCourseCreated: {
    icon: 'courseCheck', color: null, // themed per-call via opts.courseColor
    title: 'New Course Created!',
    body: '{{courseName}}',
    buttonLabel: 'Learn now',
  },
  welcomeBack: {
    icon: 'atomFlask', color: 'var(--blue-bright)',
    title: 'Welcome Back\nto Kids Learning Lab!',
    body: "It's been {{days}} days since you last opened the app!",
    buttonLabel: 'Continue learning',
  },
  premiumWelcome: {
    icon: 'ribbon', color: '#F5B400', bg: '#F5B400',
    title: 'Welcome to Kids\nLearning Lab Premium',
    body: 'Boost your learning',
    list: [
      'Unlimited lessons per day', 'Unlimited games per day', 'Up to 10 courses at once!',
      'Keep up to 4 Streak Passes!', 'Live AI assistant as you learn',
      'Learn why your answer was wrong: Explain My Answer',
      'Unlimited Personalized Review: review your mistakes and weak spots.',
    ],
    buttonLabel: 'Continue',
    onGold: true, // white text/icon on the gold background, per PDF
  },
  courseSent: {
    icon: 'shareOut', color: 'var(--purple)',
    title: 'Course\nSent!',
    body: 'Course "{{courseName}}" was shared with "{{recipient}}"\n\nAsk them to go to the Learn page to view the request.',
    buttonLabel: 'Continue',
  },
  qrProcessing: {
    icon: 'scanFrame', color: 'var(--ink)',
    title: 'Processing\nQR Code…',
    body: '',
    buttonLabel: null,
  },
  accountCreating: {
    icon: null, // uses the app logo, not a drawn icon — handled specially
    color: 'var(--blue-main)',
    title: 'Creating your\naccount…',
    body: 'Creating a Kids Learning Lab account for {{email}}…',
    buttonLabel: null,
  },
  learningGamesWelcome: {
    icon: 'controller', color: 'var(--blue-main)',
    title: 'Welcome to\nLearning Games',
    body: 'Play games that help you learn your course or another topic!',
    buttonLabel: 'Continue',
  },
  streaksWelcome: {
    icon: 'flame', color: 'var(--streak-orange)',
    title: 'Streaks',
    body: 'Complete a lesson to earn a day towards your streak every day!',
    buttonLabel: 'Start a streak',
  },
  avatarCustomizeWelcome: {
    icon: 'palette', color: 'var(--blue-main)',
    title: 'Customize\nYour Avatar',
    body: 'Choose up to 2 emojis and a color to customize your avatar that your friends see! Buy extra colors or emojis in the XP Shop!',
    buttonLabel: 'Customize',
  },
  addFriendsWelcome: {
    icon: 'people', color: 'var(--blue-main)',
    title: 'Add Friends',
    body: 'Supercharge your learning with Friends! Simply scan a QR code to send a friend request!',
    buttonLabel: 'Add a friend',
  },
  badgesWelcome: {
    icon: 'starRibbon', color: 'var(--blue-main)',
    title: 'Earn Badges',
    body: 'Earn badges by building streaks, finishing lessons, and units, playing games, adding friends, and adding and completing courses. Every badge gives you an XP boost of +100 XP.',
    buttonLabel: 'Start Earning',
  },
  adaptiveDifficultyWelcome: {
    icon: 'dial', color: 'var(--green)',
    title: 'Adaptive Difficulty',
    eyebrow: 'From Kids Learning Lab',
    body: 'Adaptive Difficulty automatically makes your next lesson harder OR easier, making lessons more challenging and making them fit your level. Turn off anytime in Settings.',
    buttonLabel: 'Keep it on',
    secondaryLabel: 'Turn off',
  },
};

// Overlays that auto-dismiss on a timer instead of waiting for a tap —
// matches the PDF's "fake 3-second spinner" overlays.
const AUTO_ADVANCE_MS = {
  personalizedReviewStart: 1400,
  qrProcessing: 3000,
  accountCreating: 3000,
};

// ============================================================
// STYLE (injected once, mirrors badges.js's injectStylesOnce pattern)
// ============================================================
function injectStylesOnce() {
  if (document.getElementById('firstTimeOverlaysStyleTag')) return;
  const style = document.createElement('style');
  style.id = 'firstTimeOverlaysStyleTag';
  style.textContent = `
    .fto-icon-wrap {
      width: 88px; height: 88px; border-radius: 50%; margin: 0 auto 22px;
      display: flex; align-items: center; justify-content: center;
      background: var(--fto-bg, var(--blue-pale)); color: var(--fto-cc, var(--blue-main));
    }
    .fto-icon-wrap svg { width: 44px; height: 44px; }
    .fto-icon-wrap.on-gold { background: rgba(255,255,255,0.22); color: var(--white); }
    .fto-eyebrow {
      font-size: 12.5px; font-weight: 700; font-style: italic; color: var(--fto-cc, var(--blue-main));
      margin: -10px 0 4px;
    }
    .fto-title {
      font-size: 22px; font-weight: 800; color: var(--fto-cc, var(--blue-deep));
      white-space: pre-line; margin: 0 0 10px; line-height: 1.25;
    }
    .fto-title.on-gold { color: var(--white); }
    .fto-body {
      font-size: 14.5px; color: var(--ink-soft); line-height: 1.5; white-space: pre-line;
      margin: 0 0 6px;
    }
    .fto-body.on-gold { color: rgba(255,255,255,0.92); }
    .fto-list { list-style: none; padding: 0; margin: 14px 0 4px; text-align: left; width: 100%; }
    .fto-list li {
      font-size: 13.5px; line-height: 1.5; padding-left: 20px; position: relative; margin-bottom: 8px;
      color: rgba(255,255,255,0.95);
    }
    .fto-list li::before { content: '•'; position: absolute; left: 4px; }
    .fto-spinner {
      width: 42px; height: 42px; margin-top: 18px;
      border: 3.5px solid var(--fto-cc, var(--blue-main)); border-top-color: transparent;
      border-radius: 50%; animation: ftoSpin 0.8s linear infinite;
    }
    @keyframes ftoSpin { to { transform: rotate(360deg); } }
    .fto-secondary-btn {
      width: 100%; padding: 13px; border: none; background: none; margin-top: 8px;
      color: var(--ink-soft); font-weight: 700; font-size: 14px; cursor: pointer; font-family: var(--app-font), sans-serif;
    }
  `;
  document.head.appendChild(style);
}

// ============================================================
// OVERLAY SHELL — one shared DOM node, re-populated per call. Clones the
// kll-modal-overlay/lesson-summary-* full-pager shell (see badges.js /
// index.html) rather than the smaller floating kll-modal card, since the
// spec calls these "full pagers".
// ============================================================
let els = null;
function ensureDom() {
  if (els) return els;
  injectStylesOnce();
  const overlay = document.createElement('div');
  overlay.className = 'kll-modal-overlay lesson-summary-overlay';
  overlay.id = 'firstTimeOverlayRoot';
  overlay.style.zIndex = '9700'; // above badge celebrations (9600), so it never gets buried mid-sequence
  overlay.innerHTML = `
    <div class="lesson-summary-container" id="ftoContainer">
      <div class="lesson-summary-body">
        <div class="fto-icon-wrap" id="ftoIconWrap"></div>
        <div class="fto-eyebrow" id="ftoEyebrow" style="display:none;"></div>
        <h2 class="fto-title" id="ftoTitle"></h2>
        <p class="fto-body" id="ftoBody"></p>
        <ul class="fto-list" id="ftoList" style="display:none;"></ul>
        <div class="fto-spinner" id="ftoSpinner" style="display:none;"></div>
      </div>
      <div class="lesson-summary-footer" id="ftoFooter">
        <button class="lesson-action-btn" id="ftoPrimaryBtn"></button>
        <button class="fto-secondary-btn" id="ftoSecondaryBtn" style="display:none;"></button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);
  els = {
    overlay,
    container: overlay.querySelector('#ftoContainer'),
    iconWrap: overlay.querySelector('#ftoIconWrap'),
    eyebrow: overlay.querySelector('#ftoEyebrow'),
    title: overlay.querySelector('#ftoTitle'),
    body: overlay.querySelector('#ftoBody'),
    list: overlay.querySelector('#ftoList'),
    spinner: overlay.querySelector('#ftoSpinner'),
    footer: overlay.querySelector('#ftoFooter'),
    primaryBtn: overlay.querySelector('#ftoPrimaryBtn'),
    secondaryBtn: overlay.querySelector('#ftoSecondaryBtn'),
  };
  return els;
}

function fillTemplate(str, vars) {
  if (!str) return '';
  return str.replace(/\{\{(\w+)\}\}/g, (_, key) => (vars && vars[key] != null ? vars[key] : ''));
}

function escapeHtml(str) {
  return (str || '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ============================================================
// PUBLIC: maybeShowOverlay
// ============================================================
// Resolves immediately (no DOM shown) if this overlay was already seen —
// callers can always `await maybeShowOverlay(...)` unconditionally and
// treat it as a no-op guard, same shape as maybeShowDiagramsInfoModal in
// learn.js.
//
// opts.force — bypasses the "already seen" gate entirely (the overlay
// still marks itself seen afterward, same as any other run). Used by
// overlays that are meant to play every time they're triggered rather
// than once-ever — currently just 'personalizedReviewStart', since a
// learner should see "Starting…" every time they start a review, not
// only the first time.
export function maybeShowOverlay(id, opts = {}) {
  const def = OVERLAY_DEFS[id];
  if (!def) {
    console.warn('firstTimeOverlays: unknown overlay id', id);
    return Promise.resolve();
  }
  if (!opts.force && hasSeenOverlay(id)) return Promise.resolve();

  return new Promise((resolve) => {
    const e = ensureDom();
    const { vars, courseColor, secondaryAction } = opts;
    const accent = def.color || courseColor || 'var(--blue-main)';
    const onGold = !!def.onGold;

    e.container.style.setProperty('--cc', accent);
    e.iconWrap.style.setProperty('--fto-cc', accent);
    if (onGold) {
      e.iconWrap.style.removeProperty('--fto-bg');
    } else {
      e.iconWrap.style.setProperty('--fto-bg', hexToPale(accent));
    }
    e.iconWrap.classList.toggle('on-gold', onGold);
    e.iconWrap.innerHTML = def.icon === null ? appLogoMarkup() : (ICONS[def.icon] || '');

    if (def.eyebrow) { e.eyebrow.textContent = def.eyebrow; e.eyebrow.style.display = ''; }
    else { e.eyebrow.style.display = 'none'; }

    e.title.textContent = fillTemplate(def.title, vars);
    e.title.classList.toggle('on-gold', onGold);
    e.body.textContent = fillTemplate(def.body, vars);
    e.body.classList.toggle('on-gold', onGold);
    e.body.style.display = def.body ? '' : 'none';

    if (def.list && def.list.length) {
      e.list.innerHTML = def.list.map((li) => `<li>${escapeHtml(li)}</li>`).join('');
      e.list.style.display = '';
    } else {
      e.list.style.display = 'none';
    }

    e.overlay.style.background = onGold ? (def.bg || '#F5B400') : '';

    const isAuto = AUTO_ADVANCE_MS[id];
    e.spinner.style.display = isAuto ? '' : 'none';
    e.footer.style.display = isAuto ? 'none' : '';

    function finish() {
      e.overlay.classList.remove('show');
      e.primaryBtn.onclick = null;
      e.secondaryBtn.onclick = null;
      markSeen(id);
      resolve();
    }

    if (isAuto) {
      e.overlay.classList.add('show');
      setTimeout(finish, isAuto);
      return;
    }

    e.primaryBtn.textContent = def.buttonLabel || 'Continue';
    e.primaryBtn.onclick = finish;

    if (def.secondaryLabel) {
      e.secondaryBtn.textContent = def.secondaryLabel;
      e.secondaryBtn.style.display = '';
      e.secondaryBtn.onclick = () => {
        // "Turn off" on the Adaptive Difficulty welcome overlay just closes
        // it without changing the setting — the real off-switch (with its
        // own "Are You Sure?" confirm) lives in Settings, per the existing
        // adaptive-difficulty flow. This is only ever the first-run welcome.
        secondaryAction?.();
        finish();
      };
    } else {
      e.secondaryBtn.style.display = 'none';
    }

    e.overlay.classList.add('show');
  });
}

// Pale-tint background for the icon circle. For the app's own accent
// tokens (var(--blue-main) etc.) this just reuses the matching pale token
// so it's pixel-identical to the rest of the app. For New Course Created —
// the one overlay themed to an arbitrary, user-picked course color — there
// is no pre-made pale variant, so color-mix() derives one at render time
// (supported in all Capacitor/WebView targets this app ships to).
const PALE_TOKEN_MAP = {
  'var(--blue-main)': 'var(--blue-pale)',
  'var(--blue-bright)': 'var(--blue-pale)',
  'var(--purple)': '#F1ECFB',
  'var(--streak-orange)': '#FFF1E2',
  'var(--green)': '#E9F7EA',
};
function hexToPale(color) {
  if (PALE_TOKEN_MAP[color]) return PALE_TOKEN_MAP[color];
  if (typeof color === 'string' && color.startsWith('var(')) return 'var(--blue-pale)';
  return `color-mix(in srgb, ${color} 16%, white)`;
}

function appLogoMarkup() {
  return `<img src="assets/imgs/logo.png" alt="" style="width:56px;height:56px;border-radius:14px;" onerror="this.style.display='none'"/>`;
}