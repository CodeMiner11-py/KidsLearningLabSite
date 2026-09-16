// tutorial.js — Kids Learning Lab first-run app tour
//
// A guided spotlight walkthrough of the four tabs, shown once, right after
// a learner's very first sign-in (or first sign-in on any pre-existing
// account that's never seen it — same "missing flag always means show it"
// philosophy as firstTimeOverlays.js). Deliberately its own module, NOT
// folded into firstTimeOverlays.js: that module's overlays are all
// full-page "kll-modal-overlay/lesson-summary" pagers with a single flag
// map, while this is a persistent, click-through-blocking spotlight layer
// that has to sit ABOVE the live app UI, drive main.js's own page
// switching mid-run, and track its position across a page transition —
// different enough machinery that sharing the module would just make
// firstTimeOverlays.js harder to reason about.
//
// ============================================================
// DATA MODEL
// ============================================================
// Local only — deliberately NOT synced to Firestore. Stored under
// localStorage key `kll_tutorial_seen_{uid}`, scoped per-account so
// signing in as a different person on the same device doesn't inherit
// (or suppress) someone else's "seen" state. The tradeoff of going local:
// the tour re-appears once per device (e.g. a reinstall, a second phone,
// or clearing site data), rather than being a true one-time-ever-anywhere
// flag — an accepted tradeoff here since a spotlight walkthrough re-firing
// on a new device once in a while is harmless, whereas a Firestore round
// trip before the very first thing a signed-in user sees isn't worth it.
//
// ============================================================
// PUBLIC API
// ============================================================
//   initTutorialForUser(uid)  — call once on sign-in (mirrors every other
//                                init*ForUser in the app), loads the flag
//   resetTutorialState()      — call on sign-out
//   maybeRunAppTour()         — no-op if already seen; otherwise starts the
//                                tour a beat after Home has settled. Call
//                                AFTER refreshHome() so the stat row/shop
//                                banner it spotlights actually exist.
//
import { db } from './firebase.js';
import { doc, setDoc, increment } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";

const TOUR_XP_REWARD = 5;
// Matches main.js's own PAGE_TRANSITION_MS (340ms) plus a small buffer so
// we never measure a target's position mid-slide.
const PAGE_SWITCH_WAIT_MS = 420;

// ============================================================
// STATE
// ============================================================
let currentUid = null;
let tourSeen = false;
let tourRunning = false;
let currentStepIndex = 0;

function seenKey(uid) { return `kll_tutorial_seen_${uid}`; }

export function initTutorialForUser(uid) {
  if (!uid) return;
  currentUid = uid;
  try {
    tourSeen = localStorage.getItem(seenKey(uid)) === '1';
  } catch (err) {
    // Fail closed on "already seen" (i.e. don't show it) rather than risk
    // re-running the tour on every launch if localStorage is unavailable
    // (e.g. private browsing) — worse to nag than to occasionally miss it.
    console.warn('tutorial: failed to read local seen flag, treating as already seen', err);
    tourSeen = true;
  }
}

export function resetTutorialState() {
  currentUid = null;
  tourSeen = false;
  tourRunning = false;
}

function markTourSeen() {
  tourSeen = true;
  if (!currentUid) return;
  try {
    localStorage.setItem(seenKey(currentUid), '1');
  } catch (err) {
    console.error('tutorial: failed to persist local seen flag', err);
  }
}

// Atomic increment on the same learnProfile doc every other XP source
// writes to (see awardBadgeXp in badges.js) — can't race/clobber a
// concurrent game or badge XP write.
async function awardTourXp() {
  if (!currentUid) return;
  try {
    await setDoc(doc(db, 'users', currentUid, 'learnProfile', 'main'), { xp: increment(TOUR_XP_REWARD) }, { merge: true });
  } catch (err) {
    console.error('tutorial: XP award failed', err);
  }
}

// ============================================================
// STEPS
// ============================================================
// `page` is the bottom-nav tab (main.js's PAGE_ORDER value) the selector
// lives on — omitted for the nav buttons themselves, which are always
// visible outside the .page containers, so no switch is needed to reach
// them. Steps are ordered to match the direction the spotlight will
// actually travel (nav bar → the page it opens → back to Home → down the
// Home stat row) so it always glides forward, never jumps around.
const STEPS = [
  {
    selector: '.nav-btn[data-page="listen"]',
    title: 'Listen',
    body: "Listen to the Kids Learning Lab Podcast for kids and family.",
  },
  {
    selector: '.listen-header',
    page: 'listen',
    title: 'Science, History & Tech',
    body: 'Browse episodes here and tap one to play, or generate trivia from one.',
  },
  {
    selector: '.nav-btn[data-page="learn"]',
    title: 'Learn',
    body: 'Create your own course with AI to learn almost anything.',
  },
  {
    selector: '.learn-top-bar',
    page: 'learn',
    title: 'Streaks, Games & Courses',
    body: 'Check your streak, jump into a Learning Game, or browse your Courses — all from up here.',
  },
  {
    selector: '.nav-btn[data-page="home"]',
    title: 'Home',
    body: 'Home is your dashboard — everything about your progress lives here.',
  },
  {
    selector: '#homeStatStreak',
    page: 'home',
    title: 'Day Streak',
    body: "Come back and learn every day to keep your streak alive!",
  },
  {
    selector: '#homeStatXp',
    page: 'home',
    title: 'Total XP',
    body: 'Earn XP as you learn, play games, and finish courses.',
  },
  {
    selector: '#homeXpShopBanner',
    page: 'home',
    title: 'XP Shop',
    body: 'Spend your XP on streak passes, avatar colors and emoji, and more.',
  },
];

// ============================================================
// STYLE (injected once, mirrors badges.js/firstTimeOverlays.js's
// injectStylesOnce pattern — this module never assumes another module's
// stylesheet has already been injected)
// ============================================================
function injectStylesOnce() {
  if (document.getElementById('tutorialStyleTag')) return;
  const style = document.createElement('style');
  style.id = 'tutorialStyleTag';
  style.textContent = `
    .tour-overlay {
      position: fixed; inset: 0; z-index: 9800;
      opacity: 0; visibility: hidden; pointer-events: none;
      transition: opacity .3s ease, visibility 0s linear .3s;
    }
    .tour-overlay.show {
      opacity: 1; visibility: visible; pointer-events: auto;
      transition: opacity .3s ease;
    }
    /* The "spotlight": a transparent box whose oversized box-shadow IS the
       dimmed backdrop — the box's own rect is the only part of the screen
       left undimmed. Sliding top/left/width/height with a normal CSS
       transition is what makes the hole glide smoothly from one target to
       the next, no SVG/mask/rAF animation loop needed. */
    .tour-spotlight {
      position: fixed;
      border-radius: 16px;
      box-shadow: 0 0 0 9999px rgba(9, 22, 45, 0.74);
      transition: top .55s cubic-bezier(.65,0,.35,1), left .55s cubic-bezier(.65,0,.35,1),
                  width .55s cubic-bezier(.65,0,.35,1), height .55s cubic-bezier(.65,0,.35,1),
                  border-radius .4s ease;
      pointer-events: none;
    }
    .tour-spotlight::after {
      content: '';
      position: absolute; inset: -5px;
      border-radius: inherit;
      border: 2.5px solid var(--white);
      box-shadow: 0 0 0 3px var(--blue-bright, #4FA6FF), 0 0 22px 5px rgba(79,166,255,0.6);
      animation: tourPulse 1.7s ease-in-out infinite;
    }
    @keyframes tourPulse {
      0%, 100% { opacity: .55; transform: scale(1); }
      50% { opacity: 1; transform: scale(1.035); }
    }
    .tour-card {
      position: fixed;
      width: 280px; max-width: calc(100vw - 32px);
      background: var(--white);
      border-radius: 18px;
      padding: 18px 20px 16px;
      box-shadow: 0 20px 50px rgba(18,59,122,0.3);
      opacity: 0;
      transform: translateY(8px) scale(.97);
      transition: top .5s cubic-bezier(.65,0,.35,1), left .5s cubic-bezier(.65,0,.35,1),
                  opacity .25s ease, transform .3s cubic-bezier(.22,.9,.32,1);
      font-family: var(--app-font), sans-serif;
    }
    .tour-card.settled { opacity: 1; transform: translateY(0) scale(1); }
    .tour-card-arrow {
      position: absolute; width: 14px; height: 14px; background: var(--white);
      transform: rotate(45deg); border-radius: 3px;
    }
    .tour-eyebrow {
      font-size: 11px; font-weight: 800; color: var(--blue-main);
      letter-spacing: .04em; text-transform: uppercase; margin-bottom: 5px;
    }
    .tour-title { font-size: 16.5px; font-weight: 800; color: var(--blue-deep); margin: 0 0 6px; }
    .tour-body { font-size: 13px; color: var(--ink-soft); line-height: 1.45; margin: 0 0 14px; }
    .tour-dots { display: flex; gap: 5px; margin-bottom: 14px; }
    .tour-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--border-soft); transition: background .2s ease, transform .2s ease; }
    .tour-dot.active { background: var(--blue-main); transform: scale(1.3); }
    .tour-footer-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
    .tour-skip-btn {
      background: none; border: none; color: var(--ink-soft);
      font-weight: 700; font-size: 12.5px; cursor: pointer; padding: 6px 2px;
      font-family: inherit;
    }
    .tour-next-btn {
      background: var(--blue-main); color: var(--white); border: none;
      border-radius: 10px; padding: 9px 20px; font-weight: 800; font-size: 13.5px;
      cursor: pointer; transition: background .15s ease, transform .1s ease;
      font-family: inherit;
    }
    .tour-next-btn:hover { background: var(--blue-deep); }
    .tour-next-btn:active { transform: scale(.95); }

    /* ---- Completion celebration (clones the kll-modal-overlay/
       lesson-summary shell already used by badge unlocks and first-time
       overlays, so it matches the app's established "finished something"
       moment instead of inventing a new visual language) ---- */
    #tourCompleteOverlay { z-index: 9900; }
    .tour-complete-icon-wrap {
      position: relative; width: 92px; height: 92px; border-radius: 50%;
      margin: 0 auto 18px;
      background: linear-gradient(135deg, var(--blue-bright), var(--blue-main));
      display: flex; align-items: center; justify-content: center;
      color: var(--white); font-size: 44px;
      animation: tourIconPop .5s cubic-bezier(.34,1.56,.64,1) both;
    }
    @keyframes tourIconPop { from { transform: scale(.4); opacity: 0; } to { transform: scale(1); opacity: 1; } }
    .tour-spark {
      position: absolute; width: 8px; height: 8px; border-radius: 50%;
      opacity: 0; animation: tourSparkPop 1.1s ease-out infinite;
    }
    .tour-spark-1 { top: -6px; left: 6px; background: var(--streak-orange); animation-delay: .35s; }
    .tour-spark-2 { top: 12px; right: -8px; background: var(--green); animation-delay: .55s; }
    .tour-spark-3 { bottom: -2px; left: 22%; background: var(--purple); animation-delay: .75s; }
    @keyframes tourSparkPop {
      0% { opacity: 0; transform: scale(.4) translateY(0); }
      35% { opacity: 1; transform: scale(1) translateY(-6px); }
      100% { opacity: 0; transform: scale(.6) translateY(-18px); }
    }
    .tour-complete-eyebrow {
      font-weight: 800; font-size: 12px; color: var(--blue-main);
      text-transform: uppercase; letter-spacing: .06em; margin-bottom: 6px; text-align: center;
    }
    .tour-complete-title { font-size: 21px; font-weight: 800; color: var(--blue-deep); margin: 0 0 8px; text-align: center; }
    .tour-complete-body { font-size: 13.5px; color: var(--ink-soft); line-height: 1.45; margin-bottom: 20px; text-align: center; }
    .tour-complete-xp {
      display: inline-flex; align-items: center; gap: 5px;
      background: var(--blue-pale); color: var(--blue-main);
      border-radius: 999px; padding: 7px 16px; font-weight: 800; font-size: 14px;
    }
    .tour-complete-xp .material-symbols-outlined { font-size: 17px; }
  `;
  document.head.appendChild(style);
}

// ============================================================
// SPOTLIGHT DOM (one shared node, lazily built — same pattern as
// firstTimeOverlays.js's ensureDom)
// ============================================================
let els = null;
function ensureDom() {
  if (els) return els;
  injectStylesOnce();

  const overlay = document.createElement('div');
  overlay.className = 'tour-overlay';
  overlay.id = 'tourOverlay';

  const dotsHtml = STEPS.map(() => '<div class="tour-dot"></div>').join('');
  overlay.innerHTML = `
    <div class="tour-spotlight" id="tourSpotlight"></div>
    <div class="tour-card" id="tourCard">
      <div class="tour-card-arrow" id="tourCardArrow"></div>
      <div class="tour-eyebrow" id="tourEyebrow"></div>
      <h3 class="tour-title" id="tourTitle"></h3>
      <p class="tour-body" id="tourBody"></p>
      <div class="tour-dots" id="tourDots">${dotsHtml}</div>
      <div class="tour-footer-row">
        <button type="button" class="tour-skip-btn" id="tourSkipBtn">Skip tour</button>
        <button type="button" class="tour-next-btn" id="tourNextBtn">Next</button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);

  els = {
    overlay,
    spotlight: overlay.querySelector('#tourSpotlight'),
    card: overlay.querySelector('#tourCard'),
    arrow: overlay.querySelector('#tourCardArrow'),
    eyebrow: overlay.querySelector('#tourEyebrow'),
    title: overlay.querySelector('#tourTitle'),
    body: overlay.querySelector('#tourBody'),
    dots: Array.from(overlay.querySelectorAll('.tour-dot')),
    skipBtn: overlay.querySelector('#tourSkipBtn'),
    nextBtn: overlay.querySelector('#tourNextBtn'),
  };

  els.skipBtn.addEventListener('click', () => finishTour(false));
  els.nextBtn.addEventListener('click', () => {
    if (currentStepIndex + 1 < STEPS.length) runStep(currentStepIndex + 1);
    else finishTour(true);
  });

  return els;
}

// ============================================================
// COMPLETION MODAL DOM
// ============================================================
let compEls = null;
function ensureCompletionDom() {
  if (compEls) return compEls;
  injectStylesOnce();

  const overlay = document.createElement('div');
  overlay.className = 'kll-modal-overlay lesson-summary-overlay';
  overlay.id = 'tourCompleteOverlay';
  overlay.innerHTML = `
    <div class="lesson-summary-container">
      <div class="lesson-summary-body">
        <div class="tour-complete-icon-wrap">
          <span class="material-symbols-outlined">celebration</span>
          <span class="tour-spark tour-spark-1"></span>
          <span class="tour-spark tour-spark-2"></span>
          <span class="tour-spark tour-spark-3"></span>
        </div>
        <div class="tour-complete-eyebrow">Tutorial Complete!</div>
        <h2 class="tour-complete-title">You're all set!</h2>
        <p class="tour-complete-body">You know your way around Kids Learning Lab now. Time to start learning.</p>
        <div class="tour-complete-xp"><span class="material-symbols-outlined">bolt</span> +${TOUR_XP_REWARD} XP</div>
      </div>
      <div class="lesson-summary-footer">
        <button type="button" class="lesson-action-btn" id="tourCompleteBtn">Let's Go!</button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);

  compEls = { overlay, btn: overlay.querySelector('#tourCompleteBtn') };
  compEls.btn.addEventListener('click', () => compEls.overlay.classList.remove('show'));
  return compEls;
}

function showCompletionModal() {
  const e = ensureCompletionDom();
  // Re-trigger the icon-pop/spark animations every time this shows, not
  // just the first — restart via a reflow-forcing class toggle.
  e.overlay.classList.remove('show');
  void e.overlay.offsetWidth;
  e.overlay.classList.add('show');
}

// ============================================================
// POSITIONING
// ============================================================
const SPOTLIGHT_PAD = 8;

function positionSpotlight(rect) {
  els.spotlight.style.top = `${rect.top - SPOTLIGHT_PAD}px`;
  els.spotlight.style.left = `${rect.left - SPOTLIGHT_PAD}px`;
  els.spotlight.style.width = `${rect.width + SPOTLIGHT_PAD * 2}px`;
  els.spotlight.style.height = `${rect.height + SPOTLIGHT_PAD * 2}px`;
  // Scale the corner rounding off the shorter side so a small square nav
  // icon reads as a soft circle while a wide banner stays a rounded rect.
  const r = Math.min(20, Math.min(rect.width, rect.height) / 2 + SPOTLIGHT_PAD);
  els.spotlight.style.borderRadius = `${r}px`;
}

// Returns true if the card was placed ABOVE the target (so the caller can
// point the little arrow the right way).
function positionCard(rect) {
  const card = els.card;
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const gap = 16;

  // Measure off-screen first so a content change (new title/body length)
  // doesn't visibly flash at the old size/position before settling.
  card.style.visibility = 'hidden';
  card.style.top = '0px';
  card.style.left = '0px';
  void card.offsetWidth;
  const cardW = card.offsetWidth;
  const cardH = card.offsetHeight;

  const spaceBelow = vh - rect.bottom;
  const spaceAbove = rect.top;
  const placeAbove = spaceBelow < cardH + gap + 12 && spaceAbove > spaceBelow;

  const top = placeAbove
    ? Math.max(rect.top - cardH - gap, 12)
    : Math.min(rect.bottom + gap, vh - cardH - 12);

  let left = rect.left + rect.width / 2 - cardW / 2;
  left = Math.max(16, Math.min(left, vw - cardW - 16));

  card.style.top = `${top}px`;
  card.style.left = `${left}px`;
  card.style.visibility = '';

  const arrowLeft = Math.max(14, Math.min(rect.left + rect.width / 2 - left - 7, cardW - 26));
  els.arrow.style.left = `${arrowLeft}px`;
  if (placeAbove) {
    els.arrow.style.top = 'auto';
    els.arrow.style.bottom = '-6px';
  } else {
    els.arrow.style.top = '-6px';
    els.arrow.style.bottom = 'auto';
  }
}

// ============================================================
// STEP FLOW
// ============================================================
function wait(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }
function nextFrame() { return new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))); }

// Drives main.js's own bottom-nav click handler so the tour reuses the
// app's real slide transition instead of a second, competing one — then
// waits for it to finish before anything measures the new page.
async function goToStepPage(targetPage) {
  if (!targetPage) return;
  const current = document.querySelector('.nav-btn.active')?.dataset.page;
  if (current === targetPage) return;
  const btn = document.querySelector(`.nav-btn[data-page="${targetPage}"]`);
  if (!btn) return;
  btn.click();
  await wait(PAGE_SWITCH_WAIT_MS);
}

async function runStep(index) {
  currentStepIndex = index;
  const def = STEPS[index];

  if (def.page) await goToStepPage(def.page);
  await nextFrame(); // let any page-switch layout settle before measuring

  const target = document.querySelector(def.selector);
  if (!target) {
    // Defensive: never strand the learner on a broken step if some target
    // isn't in the DOM for a given account state — just skip it.
    console.warn('tutorial: step target not found, skipping', def.selector);
    if (index + 1 < STEPS.length) return runStep(index + 1);
    return finishTour(true);
  }

  const rect = target.getBoundingClientRect();
  positionSpotlight(rect);

  els.eyebrow.textContent = `Step ${index + 1} of ${STEPS.length}`;
  els.title.textContent = def.title;
  els.body.textContent = def.body;
  els.nextBtn.textContent = index === STEPS.length - 1 ? "Finish" : 'Next';
  els.dots.forEach((d, i) => d.classList.toggle('active', i === index));

  positionCard(rect);

  // Re-trigger the settle-in animation on every step, not just the first.
  els.card.classList.remove('settled');
  void els.card.offsetWidth;
  els.card.classList.add('settled');
}

async function finishTour(completed) {
  tourRunning = false;
  els.overlay.classList.remove('show');
  markTourSeen(); // fire-and-forget — don't hold up the completion modal

  if (completed) {
    awardTourXp();
    // Let the spotlight overlay's own fade-out (300ms, see .tour-overlay
    // transition) finish before the completion modal fades in on top of
    // it, so the two don't visually collide.
    await wait(320);
    showCompletionModal();
  }
}

function startTour() {
  if (tourRunning) return;
  tourRunning = true;
  ensureDom();
  els.overlay.classList.add('show');
  runStep(0);
}

// ============================================================
// PUBLIC: maybeRunAppTour
// ============================================================
// No-op if already seen. Otherwise starts a beat after Home has visually
// settled (call this AFTER refreshHome() in main.js) — not awaited by the
// caller, so it never blocks the app shell from revealing.
export function maybeRunAppTour() {
  if (tourSeen || tourRunning) return;
  setTimeout(() => {
    if (!tourSeen) startTour(); // re-check in case it was marked seen elsewhere in the interim
  }, 700);
}