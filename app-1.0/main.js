import { signIn, signUp, resetPassword, logout } from './auth.js';
import { onAuthStateChanged } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";
import { auth, db, rtdb } from './firebase.js';
import {
  doc, getDoc, setDoc, deleteDoc, collection, query, where, orderBy, limit, getDocs, onSnapshot, updateDoc, writeBatch
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";
import {
  ref as rtdbRef, get as rtdbGet, set as rtdbSet
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-database.js";
import { getMessaging, getToken, onMessage, isSupported as isMessagingSupported } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-messaging.js";
import { updateProfile } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";
// ---- Elements: auth screen ----
import {updatePassword } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";
import { signInWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";
import { resetLearnState, ensureLearnInitialized, joinGameByCode, getLearnStreakSnapshot, getWeakSpotCount, openComboLesson, openReviewPage, getWeeklyReviewData } from './learn.js';
import { notifyUser } from './notifications.js';
import { initBadgesForUser, resetBadgesState, checkAccountBadge, checkFriendBadges, renderBadgeGridHtml } from './badges.js';
import { initFirstTimeOverlaysForUser, resetFirstTimeOverlaysState, maybeShowOverlay, hasSeenOverlay } from './firstTimeOverlays.js';
import { initPremiumForCurrentUser, resetPremiumState, isPremium, onPremiumChange, PREMIUM_LIMITS, limits } from './premium.js';
import { initShopStateForCurrentUser, resetShopState, onShopStateChange } from './shop.js';
import { startHomeMirrorForCurrentUser, stopHomeMirror } from './homeMirror.js';
import { initTutorialForUser, resetTutorialState, maybeRunAppTour } from './tutorial.js';
import { openXpShop } from './shopUI.js';
import { openPaywall } from './paywall.js';
import { renderJoinQr, scanJoinCode } from './multiplayer.js';
import { identifyScannedCode } from './qrRouting.js';
import { sendFriendRequestToUid } from './friends.js';

const authScreen = document.getElementById('auth-screen');
const appShell = document.getElementById('app-shell');

// ---- One-time startup video (first login on this device) ----
// Plays once ever per device (any account), then fades into the app shell.
// Locked: no controls, no pause/seek/fast-forward, no fullscreen/PiP escape.
const STARTUP_VIDEO_SEEN_KEY = 'kll_startup_video_seen';
const startupVideoOverlay = document.getElementById('startupVideoOverlay');
const startupVideo = document.getElementById('startupVideo');
let startupVideoLockedTime = 0;
let isStartupVideoShowing = false; // explicit flag — don't infer visibility from style.display

function shouldPlayStartupVideo() {
  try {
    return !localStorage.getItem(STARTUP_VIDEO_SEEN_KEY);
  } catch {
    return false; // if storage is unavailable for some reason, fail open (don't block login)
  }
}

function finishStartupVideo() {
  if (!startupVideoOverlay || !startupVideo) return;
  isStartupVideoShowing = false;
  startupVideoOverlay.style.opacity = '0';
  setTimeout(() => {
    startupVideoOverlay.style.display = 'none';
    startupVideo.pause();
    startupVideo.removeAttribute('src');
    startupVideo.load(); // release the decoded video from memory once it's done
  }, 650); // slightly longer than the CSS opacity transition so it fully fades first
}

function playStartupVideo() {
  if (!startupVideoOverlay || !startupVideo) return;

  try {
    localStorage.setItem(STARTUP_VIDEO_SEEN_KEY, '1');
  } catch {
    // ignore — worst case it plays again on this device, which is harmless
  }

  isStartupVideoShowing = true;
  startupVideoOverlay.style.display = 'flex';
  startupVideoOverlay.style.opacity = '1';
  startupVideo.currentTime = 0;
  startupVideoLockedTime = 0;

  startupVideo.play().catch(() => {
    // Autoplay-with-sound blocked by the platform — retry muted so the
    // intro still runs rather than silently never starting.
    startupVideo.muted = true;
    startupVideo.play().catch(() => finishStartupVideo()); // still blocked — skip straight in
  });
}

if (startupVideo) {
  startupVideo.addEventListener('ended', finishStartupVideo);

  // Keep track of the last legitimate playback position so any attempt to
  // seek (scrub, keyboard, programmatic) can be snapped straight back.
  startupVideo.addEventListener('timeupdate', () => {
    startupVideoLockedTime = startupVideo.currentTime;
  });
  startupVideo.addEventListener('seeking', () => {
    if (Math.abs(startupVideo.currentTime - startupVideoLockedTime) > 0.35) {
      startupVideo.currentTime = startupVideoLockedTime;
    }
  });

  // If anything manages to pause it before it's actually finished, resume
  // immediately — the only way this video stops is by playing to the end.
  startupVideo.addEventListener('pause', () => {
    if (isStartupVideoShowing && !startupVideo.ended) {
      startupVideo.play().catch(() => {});
    }
  });

  startupVideo.addEventListener('contextmenu', (e) => e.preventDefault());

  // Block keyboard shortcuts that could pause/seek/fullscreen it (space,
  // arrows, "f", "k", etc.) while the overlay is showing.
  document.addEventListener('keydown', (e) => {
    if (isStartupVideoShowing) {
      e.preventDefault();
      e.stopPropagation();
    }
  }, true);
}

const tabSignIn = document.getElementById('tab-signin');
const tabSignUp = document.getElementById('tab-signup');
const form = document.getElementById('auth-form');
const submitBtn = document.getElementById('submit-btn');
const forgotWrap = document.getElementById('forgot-wrap');
const forgotBtn = document.getElementById('forgot-btn');
const messageEl = document.getElementById('message');
const emailInput = document.getElementById('email');
const passwordInput = document.getElementById('password');

const Haptics = window.Capacitor?.Plugins?.Haptics;

// ---- Haptics settings (Profile > Set Up Haptics) ----
// Haptics default to on, at HEAVY intensity, matching the app's original
// hardcoded behavior — existing users see no change until they open the
// haptics modal and adjust it themselves.
const HAPTICS_ENABLED_KEY = 'kll_haptics_enabled';
const HAPTICS_INTENSITY_KEY = 'kll_haptics_intensity';

function getHapticsEnabled() {
  const stored = localStorage.getItem(HAPTICS_ENABLED_KEY);
  return stored === null ? true : stored === '1';
}
function setHapticsEnabled(enabled) {
  localStorage.setItem(HAPTICS_ENABLED_KEY, enabled ? '1' : '0');
}
function getHapticsIntensity() {
  return localStorage.getItem(HAPTICS_INTENSITY_KEY) || 'HEAVY';
}
function setHapticsIntensity(intensity) {
  localStorage.setItem(HAPTICS_INTENSITY_KEY, intensity);
}

// ---- Dark Mode (Profile > Additional Settings) ----
// Applied immediately below (before the splash screen even hides) so the
// app never flashes the default light theme before switching over.
// (The font switcher that used to live alongside this was removed — the
// app font is just SF Pro Display now, set directly in index.html's :root
// via --app-font, no runtime switching needed.)
const DARK_MODE_KEY = 'kll_dark_mode';

function getDarkModeEnabled() {
  try { return localStorage.getItem(DARK_MODE_KEY) === '1'; } catch { return false; }
}
function applyDarkMode(enabled) {
  document.body.classList.toggle('dark-mode', enabled);
}
function setDarkModeEnabled(enabled) {
  try { localStorage.setItem(DARK_MODE_KEY, enabled ? '1' : '0'); } catch {}
  applyDarkMode(enabled);
}
applyDarkMode(getDarkModeEnabled());

const splashScreen = document.getElementById('splash-screen');
const fontWaitModal = document.getElementById('fontWaitModal');

// ---- Splash/loading text as cached images (avoids a fallback-font flash) ----
// The very first time the app ever runs, "Loading app…" and "Loading Kids
// Learning Lab..." must render as real DOM text before SF Pro Display has
// necessarily finished loading, so that first run can briefly show the
// system fallback font. Once fonts are confirmed loaded (document.fonts.ready),
// we render both strings to an offscreen canvas in SF Pro Display, snapshot
// them as PNG data-URLs, and cache those in localStorage. Every subsequent
// launch swaps the real text out for the cached image immediately — no font
// dependency at all, so there's nothing left to flash.
const SPLASH_TEXT_CACHE_KEY = 'kll_splash_text_images_v1';

function paintSplashTextToCanvas(text, color) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const fontSize = 14;
  const dpr = window.devicePixelRatio || 1;
  ctx.font = `600 ${fontSize}px "SF Pro Display", sans-serif`;
  const width = Math.ceil(ctx.measureText(text).width) + 4;
  const height = Math.ceil(fontSize * 1.4);
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.scale(dpr, dpr);
  ctx.font = `600 ${fontSize}px "SF Pro Display", sans-serif`;
  ctx.fillStyle = color;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'left';
  ctx.fillText(text, 2, height / 2);
  return canvas.toDataURL('image/png');
}

function applyCachedSplashTextImages() {
  let cache = null;
  try { cache = JSON.parse(localStorage.getItem(SPLASH_TEXT_CACHE_KEY) || 'null'); } catch {}
  if (!cache) return false;

  const splashImg = document.getElementById('splashTextImg');
  const splashLabel = document.getElementById('splashTextLabel');
  const fontWaitImg = document.getElementById('fontWaitTextImg');
  const fontWaitLabel = document.getElementById('fontWaitTextLabel');

  if (cache.loadingApp && splashImg) {
    splashImg.src = cache.loadingApp;
    splashImg.style.display = 'block';
    if (splashLabel) splashLabel.style.display = 'none';
  }
  if (cache.loadingKll && fontWaitImg) {
    fontWaitImg.src = cache.loadingKll;
    fontWaitImg.style.display = 'block';
    if (fontWaitLabel) fontWaitLabel.style.display = 'none';
  }
  return !!(cache.loadingApp && cache.loadingKll);
}

function generateAndCacheSplashTextImages() {
  // ink-soft, matching .splash-text's CSS color — kept as a literal here since
  // canvas can't read a CSS var directly.
  const inkSoft = getComputedStyle(document.documentElement).getPropertyValue('--ink-soft').trim() || '#6B7A99';
  try {
    const cache = {
      loadingApp: paintSplashTextToCanvas('Loading app…', inkSoft),
      loadingKll: paintSplashTextToCanvas('Loading Kids Learning Lab...', inkSoft),
    };
    localStorage.setItem(SPLASH_TEXT_CACHE_KEY, JSON.stringify(cache));
    applyCachedSplashTextImages();
  } catch (err) {
    // Cache write failed (storage full/unavailable) — plain text labels stay
    // visible, which is a safe fallback, just not a cached one.
    console.error('[splashTextCache] failed to generate/cache:', err.message);
  }
}

// Try the cache immediately (covers every run after the first). If nothing
// is cached yet, this is the first-ever load — fall through to plain text
// for now, and generate the cache in the background once fonts are ready.
const hadFullSplashTextCache = applyCachedSplashTextImages();
if (!hadFullSplashTextCache && document.fonts) {
  document.fonts.ready.then(generateAndCacheSplashTextImages);
}

// The 3s splash always plays out in full first. Once it ends, if the app's
// custom fonts (SF Pro Display/Text, self-hosted as .otf — see the
// @font-face rules at the top of index.html) haven't finished loading yet,
// swap to an uncancellable "Loading Kids Learning Lab..." modal instead of
// letting the splash disappear onto unstyled/fallback-font content. That
// modal has no close button and no backdrop dismiss — it hides itself the
// instant document.fonts.ready resolves, however long that takes.
setTimeout(() => {
  if (document.fonts && document.fonts.status !== 'loaded') {
    fontWaitModal.classList.add('show');
    document.fonts.ready.then(() => {
      fontWaitModal.classList.remove('show');
      splashScreen.style.display = 'none';
      maybeRunRobotCheck();
    });
  } else {
    splashScreen.style.display = 'none';
    maybeRunRobotCheck();
  }
}, 3000);

// ============================================================
// SECURITY: random "robot" (slider) verification — ~5% of app opens
// ============================================================
const robotVerifyModalOverlay = document.getElementById('robotVerifyModalOverlay');
const robotSliderTrack = document.getElementById('robotSliderTrack');
const robotSliderHandle = document.getElementById('robotSliderHandle');
const robotSliderFill = document.getElementById('robotSliderFill');
const robotSliderLabel = document.getElementById('robotSliderLabel');
const robotLoadingWrap = document.getElementById('robotLoadingWrap');
const robotLoadingFill = document.getElementById('robotLoadingFill');

function runRobotSlider(onComplete) {
  if (!robotVerifyModalOverlay || !robotSliderTrack || !robotSliderHandle) { onComplete(); return; }

  let maxX = 0;
  let currentX = 0;
  let dragOffset = 0;
  let dragging = false;
  let completed = false;

  function computeMax() {
    maxX = Math.max(0, robotSliderTrack.clientWidth - robotSliderHandle.offsetWidth - 6);
  }
  function applyX(x) {
    currentX = Math.max(0, Math.min(maxX, x));
    robotSliderHandle.style.transform = `translateX(${currentX}px)`;
    robotSliderFill.style.width = `${currentX + robotSliderHandle.offsetWidth}px`;
    robotSliderLabel.style.opacity = String(Math.max(0, 1 - currentX / (maxX || 1)));
  }
  function resetSlider() {
    computeMax();
    applyX(0);
  }
  function startLoadingSequence() {
    completed = true;
    robotSliderTrack.style.pointerEvents = 'none';
    robotLoadingWrap.style.display = 'block';
    robotLoadingFill.style.transition = 'none';
    robotLoadingFill.style.width = '0%';
    requestAnimationFrame(() => {
      robotLoadingFill.style.transition = 'width 3s linear';
      robotLoadingFill.style.width = '100%';
    });
    setTimeout(() => {
      robotVerifyModalOverlay.classList.remove('show');
      cleanup();
      onComplete();
    }, 3050);
  }
  function onDown(e) {
    if (completed) return;
    dragging = true;
    const clientX = e.clientX ?? e.touches?.[0]?.clientX ?? 0;
    dragOffset = clientX - currentX;
    robotSliderHandle.setPointerCapture?.(e.pointerId);
  }
  function onMove(e) {
    if (!dragging || completed) return;
    const clientX = e.clientX ?? e.touches?.[0]?.clientX ?? 0;
    applyX(clientX - dragOffset);
  }
  function onUp() {
    if (!dragging || completed) return;
    dragging = false;
    if (currentX >= maxX - 2) {
      startLoadingSequence();
    } else {
      applyX(0);
    }
  }
  function cleanup() {
    robotSliderHandle.removeEventListener('pointerdown', onDown);
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', onUp);
  }

  robotSliderHandle.addEventListener('pointerdown', onDown);
  window.addEventListener('pointermove', onMove);
  window.addEventListener('pointerup', onUp);

  robotSliderTrack.style.pointerEvents = '';
  robotLoadingWrap.style.display = 'none';
  resetSlider();
  robotVerifyModalOverlay.classList.add('show');
}

function maybeRunRobotCheck() {
  return new Promise((resolve) => {
    if (Math.random() < 0.05) {
      runRobotSlider(resolve);
    } else {
      resolve();
    }
  });
}


// ============================================================
// SECURITY: login-rate security check (>25 sign-ins/hour)
// ============================================================
const LOGIN_RATE_WINDOW_MS = 60 * 60 * 1000; // 1 hour
const LOGIN_RATE_LIMIT = 100000; // more sign-ins than this within the window triggers a security check
const LOGIN_ATTEMPTS_KEY = 'kll_login_attempts';

// Rolling window of successful sign-in timestamps on this device, used to
// detect an unusually high rate of sign-ins to a single account.
function recordLoginAttempt() {
  let attempts = [];
  try { attempts = JSON.parse(localStorage.getItem(LOGIN_ATTEMPTS_KEY) || '[]'); } catch { attempts = []; }
  const now = Date.now();
  attempts = attempts.filter((t) => now - t < LOGIN_RATE_WINDOW_MS);
  attempts.push(now);
  try { localStorage.setItem(LOGIN_ATTEMPTS_KEY, JSON.stringify(attempts)); } catch {}
  return attempts.length;
}

document.addEventListener('pointerdown', (e) => {
  if (e.target.closest('button, .tab, .nav-btn') && getHapticsEnabled()) {

    Haptics?.impact({ style: getHapticsIntensity() });

  }
});

document.getElementById("quitAppBtn")?.addEventListener("click", () => {
  window.location.reload();
});

// ---- Haptics settings modal (Profile page, below Badges) ----
const hapticsSetupBtn = document.getElementById('hapticsSetupBtn');
const hapticsModalOverlay = document.getElementById('hapticsModalOverlay');
const hapticsOnOffSwitch = document.getElementById('hapticsOnOffSwitch');
const hapticsIntensityWrap = document.getElementById('hapticsIntensityWrap');
const hapticsIntensityBtns = document.querySelectorAll('.haptics-intensity-btn');
const hapticsModalDoneBtn = document.getElementById('hapticsModalDoneBtn');

function renderHapticsModal() {
  const enabled = getHapticsEnabled();
  const intensity = getHapticsIntensity();
  hapticsOnOffSwitch?.classList.toggle('on', enabled);
  hapticsOnOffSwitch?.setAttribute('aria-checked', String(enabled));
  hapticsIntensityWrap?.classList.toggle('hidden', !enabled);
  hapticsIntensityBtns.forEach((b) => {
    b.classList.toggle('active', b.dataset.intensity === intensity);
  });
}

hapticsSetupBtn?.addEventListener('click', () => {
  renderHapticsModal();
  hapticsModalOverlay?.classList.add('show');
});

hapticsOnOffSwitch?.addEventListener('click', () => {
  setHapticsEnabled(!getHapticsEnabled());
  renderHapticsModal();
});

hapticsIntensityBtns.forEach((btn) => {
  btn.addEventListener('click', () => {
    setHapticsIntensity(btn.dataset.intensity);
    renderHapticsModal();
  });
});

hapticsModalDoneBtn?.addEventListener('click', () => {
  hapticsModalOverlay?.classList.remove('show');
});

hapticsModalOverlay?.addEventListener('click', (e) => {
  if (e.target === hapticsModalOverlay) hapticsModalOverlay.classList.remove('show');
});

// ---- Dark Mode switch (Profile page, below Set Up Haptics) ----
const darkModeSwitch = document.getElementById('darkModeSwitch');

function renderDarkModeSwitch() {
  const enabled = getDarkModeEnabled();
  darkModeSwitch?.classList.toggle('on', enabled);
  darkModeSwitch?.setAttribute('aria-checked', String(enabled));
}
renderDarkModeSwitch();

darkModeSwitch?.addEventListener('click', () => {
  setDarkModeEnabled(!getDarkModeEnabled());
  renderDarkModeSwitch();
});

// ---- Diagrams: SVG/Pexels toggle (Profile page, below Reload App) ----
// Beta feature flag, OFF by default — when off, learn.js skips rendering
// any lesson diagram (Pexels photo, template SVG, or raw SVG fallback)
// regardless of what the worker returned for that part.
const DIAGRAMS_ENABLED_KEY = 'kll_diagrams_enabled';
const diagramsToggleSwitch = document.getElementById('diagramsToggleSwitch');

function getDiagramsEnabled() {
  // Default ON — an unset value (new install, or a device that's never
  // touched the toggle) reads as enabled. Only an explicit '0' turns it off.
  try { return localStorage.getItem(DIAGRAMS_ENABLED_KEY) !== '0'; } catch { return true; }
}
function setDiagramsEnabled(enabled) {
  try { localStorage.setItem(DIAGRAMS_ENABLED_KEY, enabled ? '1' : '0'); } catch {}
}
function renderDiagramsToggleSwitch() {
  const enabled = getDiagramsEnabled();
  diagramsToggleSwitch?.classList.toggle('on', enabled);
  diagramsToggleSwitch?.setAttribute('aria-checked', String(enabled));
}
renderDiagramsToggleSwitch();

diagramsToggleSwitch?.addEventListener('click', () => {
  setDiagramsEnabled(!getDiagramsEnabled());
  renderDiagramsToggleSwitch();
});

// ---- Adaptive Difficulty toggle (Profile page, below Diagrams) ----
// ON by default (unlike the diagrams beta flag above) — a missing/unset
// localStorage value reads as enabled. Reads the score of the last lesson
// completed in the active course (see learn.js's finishLesson()) and shifts
// the NEXT lesson's difficulty up/down/steady accordingly. Turning it off
// asks for confirmation, since it's a genuinely useful feature most
// learners benefit from keeping on.
const ADAPTIVE_DIFFICULTY_ENABLED_KEY = 'kll_adaptive_difficulty_enabled';
const adaptiveDifficultyToggleSwitch = document.getElementById('adaptiveDifficultyToggleSwitch');

function getAdaptiveDifficultyEnabled() {
  try {
    const stored = localStorage.getItem(ADAPTIVE_DIFFICULTY_ENABLED_KEY);
    return stored === null ? true : stored === '1';
  } catch { return true; }
}
function setAdaptiveDifficultyEnabled(enabled) {
  try { localStorage.setItem(ADAPTIVE_DIFFICULTY_ENABLED_KEY, enabled ? '1' : '0'); } catch {}
}
function renderAdaptiveDifficultyToggleSwitch() {
  const enabled = getAdaptiveDifficultyEnabled();
  adaptiveDifficultyToggleSwitch?.classList.toggle('on', enabled);
  adaptiveDifficultyToggleSwitch?.setAttribute('aria-checked', String(enabled));
}
renderAdaptiveDifficultyToggleSwitch();

const adaptiveDifficultyOffModalOverlay = document.getElementById('adaptiveDifficultyOffModalOverlay');
const adaptiveDifficultyOffCancelBtn = document.getElementById('adaptiveDifficultyOffCancelBtn');
const adaptiveDifficultyOffConfirmBtn = document.getElementById('adaptiveDifficultyOffConfirmBtn');

adaptiveDifficultyToggleSwitch?.addEventListener('click', () => {
  const currentlyEnabled = getAdaptiveDifficultyEnabled();
  if (currentlyEnabled) {
    if (!adaptiveDifficultyOffModalOverlay) return; // no modal in DOM — fail safe, don't turn it off silently
    adaptiveDifficultyOffModalOverlay.classList.add('show');
    return;
  }
  setAdaptiveDifficultyEnabled(true);
  renderAdaptiveDifficultyToggleSwitch();
});

adaptiveDifficultyOffCancelBtn?.addEventListener('click', () => {
  adaptiveDifficultyOffModalOverlay.classList.remove('show');
});

adaptiveDifficultyOffConfirmBtn?.addEventListener('click', () => {
  setAdaptiveDifficultyEnabled(false);
  renderAdaptiveDifficultyToggleSwitch();
  adaptiveDifficultyOffModalOverlay.classList.remove('show');
});

// ---- Review Lessons toggle (Profile page, below Adaptive Difficulty) ----
// ON by default. Off: hides the Review Page entry points (Home button +
// course-header button, see learn.js's updateReviewWrongAnswersBtn/
// openReviewPage) and stops the end-of-lesson AI weak/strong-spot
// generation entirely.
const REVIEW_LESSONS_ENABLED_KEY = 'kll_review_lessons_enabled';
const reviewLessonsToggleSwitch = document.getElementById('reviewLessonsToggleSwitch');

function getReviewLessonsToggleEnabled() {
  try {
    const stored = localStorage.getItem(REVIEW_LESSONS_ENABLED_KEY);
    return stored === null ? true : stored === '1';
  } catch { return true; }
}
function setReviewLessonsToggleEnabled(enabled) {
  try { localStorage.setItem(REVIEW_LESSONS_ENABLED_KEY, enabled ? '1' : '0'); } catch {}
}
function renderReviewLessonsToggleSwitch() {
  const enabled = getReviewLessonsToggleEnabled();
  reviewLessonsToggleSwitch?.classList.toggle('on', enabled);
  reviewLessonsToggleSwitch?.setAttribute('aria-checked', String(enabled));
  const reviewBtn = document.getElementById('homeReviewPageBtn');
  if (reviewBtn) reviewBtn.style.display = enabled ? '' : 'none';
}
renderReviewLessonsToggleSwitch();

reviewLessonsToggleSwitch?.addEventListener('click', () => {
  setReviewLessonsToggleEnabled(!getReviewLessonsToggleEnabled());
  renderReviewLessonsToggleSwitch();
});

// ---- Turn Off Premium Features (Profile > Additional Settings) ----
// A self-service downgrade for THIS account only — sets premium:false on
// its own Firestore doc, same field the RevenueCat purchase flow sets to
// true. It never touches the actual App Store purchase/entitlement, so
// nothing is refunded or lost — Restore Purchases brings it right back.
// Only shown at all when the signed-in account is currently premium (see
// the onPremiumChange listener below), since there's nothing to turn off
// otherwise.
const turnOffPremiumRow = document.getElementById('turnOffPremiumRow');
const turnOffPremiumBtn = document.getElementById('turnOffPremiumBtn');
const turnOffPremiumModalOverlay = document.getElementById('turnOffPremiumModalOverlay');
const turnOffPremiumCancelBtn = document.getElementById('turnOffPremiumCancelBtn');
const turnOffPremiumConfirmBtn = document.getElementById('turnOffPremiumConfirmBtn');

onPremiumChange((premium) => {
  if (turnOffPremiumRow) turnOffPremiumRow.style.display = premium ? '' : 'none';
});

turnOffPremiumBtn?.addEventListener('click', () => {
  if (!turnOffPremiumModalOverlay) return; // no modal in DOM — fail safe, don't turn it off silently
  turnOffPremiumModalOverlay.classList.add('show');
});

turnOffPremiumCancelBtn?.addEventListener('click', () => {
  turnOffPremiumModalOverlay.classList.remove('show');
});

turnOffPremiumConfirmBtn?.addEventListener('click', async () => {
  const u = auth.currentUser;
  if (!u) return;
  turnOffPremiumConfirmBtn.disabled = true;
  turnOffPremiumConfirmBtn.textContent = 'Turning off…';
  try {
    await setDoc(doc(db, 'users', u.uid, 'learnProfile', 'main'), { premium: false }, { merge: true });
    // Same reasoning as the purchase-success reload in paywall.js: premium
    // state gates a lot of already-mounted logic (daily limits, the AI
    // Assistant FAB, course-slot math, RevenueCat's own cached
    // entitlement check) that's far simpler to let re-initialize fresh
    // than to try to live-patch back down.
    window.location.reload();
  } catch (err) {
    console.error('Failed to turn off premium:', err);
    turnOffPremiumConfirmBtn.disabled = false;
    turnOffPremiumConfirmBtn.textContent = 'Turn Off Premium';
    turnOffPremiumModalOverlay.classList.remove('show');
  }
});

// ---- Profile page ----
const settingsAvatar = document.getElementById('settingsAvatar');
const settingsName = document.getElementById('settingsName');
const settingsEmail = document.getElementById('settingsEmail');
const settingsStatus = document.getElementById('settingsStatus');
const settingsSignOutBtn = document.getElementById('settingsSignOutBtn');
const settingsResetPwBtn = document.getElementById('settingsResetPwBtn');
const settingsDeleteBtn = document.getElementById('settingsDeleteBtn');

const profileXpCount = document.getElementById('profileXpCount');
const profileXpRowBtn = document.getElementById('profileXpRowBtn');
profileXpRowBtn?.addEventListener('click', () => openXpShop());
const profileChangeNameBtn = document.getElementById('profileChangeNameBtn');
const profileAvatarBtn = document.getElementById('profileAvatarBtn');
const profileFriendsBtn = document.getElementById('profileFriendsBtn');
const profileAccountSettingsBtn = document.getElementById('profileAccountSettingsBtn');

const accountSettingsPageOverlay = document.getElementById('accountSettingsPageOverlay');
const accountSettingsExitBtn = document.getElementById('accountSettingsExitBtn');

const deleteAccountModalOverlay = document.getElementById('deleteAccountModalOverlay');
const deleteAccountError = document.getElementById('deleteAccountError');
const deleteAccountConfirmBtn = document.getElementById('deleteAccountConfirmBtn');
const deleteAccountCancelBtn = document.getElementById('deleteAccountCancelBtn');

profileAccountSettingsBtn.addEventListener('click', () => {
  settingsStatus.textContent = '';
  updateSettingsUI(auth.currentUser);
  accountSettingsPageOverlay.classList.add('show');
});
accountSettingsExitBtn.addEventListener('click', () => {
  accountSettingsPageOverlay.classList.remove('show');
});

settingsDeleteBtn.addEventListener('click', () => {
  deleteAccountError.textContent = '';
  deleteAccountModalOverlay.classList.add('show');
});

deleteAccountCancelBtn.addEventListener('click', () => {
  deleteAccountModalOverlay.classList.remove('show');
});

deleteAccountConfirmBtn.addEventListener('click', async () => {
  deleteAccountConfirmBtn.disabled = true;
  deleteAccountConfirmBtn.textContent = 'Deleting…';
  try {
    const { deleteUser } = await import("https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js");
    await deleteUser(auth.currentUser);
    deleteAccountModalOverlay.classList.remove('show');
    // onAuthStateChanged handles the screen swap back to auth
  } catch (err) {
    deleteAccountError.textContent = err.code === 'auth/requires-recent-login'
      ? 'For your security, please log out and log in again to confirm deletion.'
      : (err.message || 'Could not delete account.');
  } finally {
    deleteAccountConfirmBtn.disabled = false;
    deleteAccountConfirmBtn.textContent = 'Delete Account';
  }
});

// Keeps the Profile page's name/email/avatar in sync, and lazily creates the
// public lookup/profile docs for accounts that predate the Friends/XP feature
// (self-healing — every visit here upserts them rather than requiring a
// one-time migration).
function updateSettingsUI(user) {
  if (!user) return;
  const name = user.displayName || (user.email ? user.email.split('@')[0] : 'Learner');
  settingsName.textContent = formatDisplayName(name, isPremium());
  settingsEmail.textContent = user.email || '';
  renderAvatarInto(settingsAvatar, currentAvatar); // show cached avatar immediately, no flash of the placeholder
  loadAvatarForCurrentUser(); // then always re-fetch the latest from Firestore
  syncPublicProfileDocs(user);
  loadXpTotal();
}

// ---- "Go Premium" entry points (one card per page) ----
// All share the .go-premium-entry class, so one listener + one visibility
// toggle covers every instance, wherever it lives in the DOM. The fixed
// corner pill (#goPremiumPill) is NOT one of these — see below — it never
// hides, it just changes what tapping it does once premium is active.
document.querySelectorAll('.go-premium-entry').forEach((btn) => {
  btn.addEventListener('click', () => openPaywall());
});
onPremiumChange((premium) => {
  document.querySelectorAll('.go-premium-entry').forEach((el) => {
    el.classList.toggle('is-hidden', premium);
    el.style.display = premium ? 'none' : '';
  });
});

// ---- "Go Premium" corner pill ----
// Free members: blue pill, tapping it opens the paywall (same as the
// go-premium-entry cards). Premium members: instead of disappearing, the
// pill switches to a gold "you're premium" look and tapping it opens a
// read-only benefits summary — see premiumBenefitsModalOverlay.
const goPremiumPill = document.getElementById('goPremiumPill');
const premiumBenefitsModalOverlay = document.getElementById('premiumBenefitsModalOverlay');
const premiumBenefitsList = document.getElementById('premiumBenefitsList');
const premiumBenefitsDoneBtn = document.getElementById('premiumBenefitsDoneBtn');
const premiumBenefitsExitBtn = document.getElementById('premiumBenefitsExitBtn');

const PREMIUM_BENEFIT_ITEMS = [
  { icon: 'library_books', label: `Up to ${PREMIUM_LIMITS.premium.maxCourses} courses at once` },
  { icon: 'local_fire_department', label: '4 Streak Pass slots' },
  { icon: 'auto_awesome', label: 'AI Assistant' },
  { icon: 'psychology', label: 'Explain My Answer' },
  { icon: 'history_edu', label: 'Review Page' },
  { icon: 'auto_fix_high', label: 'Combo Lessons' },
];

function renderPremiumBenefitsModal() {
  if (!premiumBenefitsList) return;
  premiumBenefitsList.innerHTML = PREMIUM_BENEFIT_ITEMS.map((item) => `
    <div class="premium-benefit-item">
      <span class="material-symbols-outlined">${item.icon}</span>
      <span class="premium-benefit-item-text">${escapeHtmlMain(item.label)}</span>
    </div>
  `).join('');
}

goPremiumPill?.addEventListener('click', () => {
  if (isPremium()) {
    renderPremiumBenefitsModal();
    premiumBenefitsModalOverlay?.classList.add('show');
  } else {
    openPaywall();
  }
});

premiumBenefitsDoneBtn?.addEventListener('click', () => {
  premiumBenefitsModalOverlay?.classList.remove('show');
});

premiumBenefitsExitBtn?.addEventListener('click', () => {
  premiumBenefitsModalOverlay?.classList.remove('show');
});

// No backdrop-tap-to-close here — this is now a full-page overlay (same
// shell as Learning Games/Friends/the paywall), not a floating card on a
// dimmed backdrop, so there's no backdrop to tap. Closes only via the
// topbar X or the Done button above.

onPremiumChange((premium) => {
  if (!goPremiumPill) return;
  goPremiumPill.classList.toggle('is-gold', premium);
  // Same icon/label either way — the gold gradient (is-gold) is what signals
  // membership; only the aria-label and click behavior change.
  goPremiumPill.setAttribute('aria-label', premium ? 'View your Premium benefits' : 'Go Premium');
});

settingsSignOutBtn.addEventListener('click', async () => {
  // Cache this account's email (no password) for the "last used account"
  // quick sign-in box, while auth.currentUser is still populated —
  // logout() below clears the session.
  saveLastUsedAccount();
  await logout();
});

settingsResetPwBtn.addEventListener('click', () => {
  if (cpElementsReady) {
    openChangePasswordModal();
  } else {
    console.warn('Change Password modal elements are missing from the DOM — add the cpModal1/2/3 markup to index.html.');
    settingsStatus.textContent = 'Change password is temporarily unavailable.';
    settingsStatus.className = 'settings-status error';
  }
});

let mode = 'signin';

function setMode(newMode) {
  mode = newMode;
  tabSignIn.classList.toggle('active', mode === 'signin');
  tabSignUp.classList.toggle('active', mode === 'signup');
  submitBtn.textContent = mode === 'signin' ? 'Sign In' : 'Create Account';
  forgotWrap.style.display = mode === 'signin' ? 'block' : 'none';
  passwordInput.setAttribute('autocomplete', mode === 'signin' ? 'current-password' : 'new-password');
  clearMessage();
}

function showMessage(text, type) {
  messageEl.textContent = text;
  messageEl.className = 'message ' + type;
}

function clearMessage() {
  messageEl.textContent = '';
  messageEl.className = 'message';
}

tabSignIn.addEventListener('click', () => setMode('signin'));
tabSignUp.addEventListener('click', () => setMode('signup'));

forgotBtn.addEventListener('click', async () => {
  const email = emailInput.value.trim();
  if (!email) {
    showMessage('Enter your email above first.', 'error');
    return;
  }
  forgotBtn.disabled = true;
  const result = await resetPassword(email);
  forgotBtn.disabled = false;
  showMessage(
    result.success ? 'Password reset email sent.' : result.error,
    result.success ? 'success' : 'error'
  );
});

// ---- Email verification + username setup ----
const RESEND_WORKER_URL = 'https://emailworkerkidslearninglabanyhtmlnonspecific.nameless-cherry-998c.workers.dev/send';
const USERNAME_ALLOWED = /^[a-zA-Z0-9 ]+$/;
// Blocks names that impersonate the [PREMIUM ⭐️] badge prefix or otherwise
// claim premium/pro status that isn't theirs — "pro" as a whole word so
// legitimate names like "Prosper" or "Prometheus" still work.
const RESERVED_NAME_PATTERN = /premium|\bpro\b/i;
const PREMIUM_NAME_PREFIX = '[PREMIUM ⭐️] ';

// Applies the premium badge prefix at render time only — the stored
// displayName itself is always just the plain name the person chose (and
// can never itself contain the prefix, see RESERVED_NAME_PATTERN above).
// That way the badge disappears immediately/automatically if premium ever
// lapses, with nothing to clean up in Firestore.
function formatDisplayName(name, premium) {
  const clean = name || 'Learner';
  return premium ? `${PREMIUM_NAME_PREFIX}${clean}` : clean;
}

// Current user's own display name, formatted the same way it would appear
// to a friend (premium prefix included) — used wherever we need to pass an
// already-formatted "from" name into the DOM-free friends.js helpers.
function myFormattedDisplayName() {
  const u = auth.currentUser;
  return formatDisplayName(u?.displayName || (u?.email ? u.email.split('@')[0] : 'Someone'), isPremium());
}

let verifyCode = null;
let verifyExpiry = null;
let verifyTimerInterval = null;
let verifyEmail = null;
let verificationInProgress = false;
// True for the entire span the first-run onboarding wizard is on screen —
// including the moment signUp() inside it fires onAuthStateChanged with a
// real user. Same purpose as verificationInProgress above: keeps that
// listener from yanking the screen to the app shell / username modal out
// from under wizard Steps 7-9, which is still showing even though a real
// account now exists.
let onboardingInProgress = false;

const verifyModalOverlay = document.getElementById('verifyModalOverlay');
const verifyCodeInput = document.getElementById('verifyCodeInput');
const verifyError = document.getElementById('verifyError');
const verifyTimer = document.getElementById('verifyTimer');
const verifySubmitBtn = document.getElementById('verifySubmitBtn');
const verifyResendBtn = document.getElementById('verifyResendBtn');

const usernameModalOverlay = document.getElementById('usernameModalOverlay');
const usernameInput = document.getElementById('usernameInput');
const usernameError = document.getElementById('usernameError');
const usernameSubmitBtn = document.getElementById('usernameSubmitBtn');

function generateCode() {
  return String(Math.floor(100000 + Math.random() * 900000));
}

async function sendVerificationEmail(email, code) {
  const html = `
    <div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:40px 32px;background:#F5FAFF;border-radius:18px;border:1.5px solid #DCE7F5">
      <img src="https://kidslearninglab.com/wp-content/uploads/2025/02/podcast-logo-app-rounded.png" style="width:48px;height:48px;border-radius:14px;display:block;margin:0 auto 20px">
      <h2 style="text-align:center;color:#14213D;margin-bottom:8px">Verify your email</h2>
      <p style="text-align:center;color:#5B6B85;font-size:14px;line-height:1.6;margin-bottom:28px">Enter this code in Kids Learning Lab to complete your sign-up. It expires in 10 minutes.</p>
      <div style="background:#fff;border:1.5px solid #DCE7F5;border-radius:14px;padding:28px;text-align:center;margin-bottom:24px">
        <span style="font-size:2.5rem;font-weight:900;letter-spacing:.25em;color:#1E6FE0">${code}</span>
      </div>
      <p style="text-align:center;color:#5B6B85;font-size:12px">If you didn't sign up for Kids Learning Lab, ignore this email.</p>
    </div>`;
  await fetch(RESEND_WORKER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ to: email, subject: 'Your Kids Learning Lab verification code', html })
  });
}

function startVerifyTimer() {
  clearInterval(verifyTimerInterval);
  verifyTimerInterval = setInterval(() => {
    const remaining = verifyExpiry - Date.now();
    if (remaining <= 0) {
      clearInterval(verifyTimerInterval);
      verifyTimer.textContent = 'Code expired. Please resend.';
      verifyCode = null;
      return;
    }
    const mins = Math.floor(remaining / 60000);
    const secs = Math.floor((remaining % 60000) / 1000);
    verifyTimer.textContent = `Expires in ${mins}:${secs.toString().padStart(2, '0')}`;
  }, 1000);
}

/* ═══════════════════════════════════════════════
   CHANGE PASSWORD — 3-STEP MODALS
   (guarded: missing DOM elements skip wiring instead
   of throwing and killing the rest of this script)
═══════════════════════════════════════════════ */
const cpModal1 = document.getElementById('cpModal1');
const cpModal2 = document.getElementById('cpModal2');
const cpModal3 = document.getElementById('cpModal3');

const cpOldPassword = document.getElementById('cpOldPassword');
const cpError1 = document.getElementById('cpError1');
const cpBtn1 = document.getElementById('cpBtn1');

const cpEmailDisplay = document.getElementById('cpEmailDisplay');
const cpSendCodeBtn = document.getElementById('cpSendCodeBtn');
const cpVerifyCodeInput = document.getElementById('cpVerifyCode');
const cpVerifyCodeBtn = document.getElementById('cpVerifyCodeBtn');
const cpError2 = document.getElementById('cpError2');

const cpNewPassword = document.getElementById('cpNewPassword');
const cpConfirmPassword = document.getElementById('cpConfirmPassword');
const cpError3 = document.getElementById('cpError3');
const cpBtn3 = document.getElementById('cpBtn3');

const cpElementsReady = !!(cpModal1 && cpModal2 && cpModal3 && cpOldPassword &&
  cpError1 && cpBtn1 && cpEmailDisplay && cpSendCodeBtn && cpVerifyCodeInput &&
  cpVerifyCodeBtn && cpError2 && cpNewPassword && cpConfirmPassword &&
  cpError3 && cpBtn3);

if (!cpElementsReady) {
  console.warn('Change Password modal elements are missing from the DOM — skipping wiring. Check index.html for the cpModal1/2/3 markup.');
}

let cpCode = null, cpCodeExpiry = null;

function cpShowError(step, msg) {
  const el = step === 1 ? cpError1 : step === 2 ? cpError2 : cpError3;
  if (el) el.textContent = msg;
}

function closeCpModal(step) {
  (step === 1 ? cpModal1 : step === 2 ? cpModal2 : cpModal3)?.classList.remove('show');
}

function openChangePasswordModal() {
  if (!cpElementsReady) return;
  cpOldPassword.value = '';
  cpShowError(1, '');
  cpBtn1.disabled = false;
  cpBtn1.textContent = 'Confirm Password';
  cpModal1.classList.add('show');
  setTimeout(() => cpOldPassword.focus(), 80);
}

if (cpElementsReady) {
  // Step 1 — reauthenticate with current password
  cpBtn1.addEventListener('click', async () => {
    const pw = cpOldPassword.value;
    cpShowError(1, '');
    if (!pw) { cpShowError(1, 'Please enter your current password.'); return; }

    cpBtn1.disabled = true;
    cpBtn1.textContent = 'Checking…';
    try {
      await signInWithEmailAndPassword(auth, auth.currentUser.email, pw);
      closeCpModal(1);

      // Reset + open step 2
      cpEmailDisplay.textContent = auth.currentUser.email || '';
      cpVerifyCodeInput.value = '';
      cpVerifyCodeInput.style.display = 'none';
      cpVerifyCodeBtn.style.display = 'none';
      cpSendCodeBtn.style.display = 'block';
      cpSendCodeBtn.disabled = false;
      cpSendCodeBtn.textContent = 'Send Code to Email';
      cpShowError(2, '');
      cpModal2.classList.add('show');
    } catch {
      cpShowError(1, 'Incorrect password. Please try again.');
      cpBtn1.disabled = false;
      cpBtn1.textContent = 'Confirm Password';
    }
  });

  // Step 2 — email a 6-digit code (reuses the existing verification-email worker)
  cpSendCodeBtn.addEventListener('click', async () => {
    cpSendCodeBtn.disabled = true;
    cpSendCodeBtn.textContent = 'Sending…';
    cpCode = generateCode(); // already defined above in main.js for sign-up verification
    cpCodeExpiry = Date.now() + 10 * 60 * 1000;

    const html = `
      <div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:40px 32px;background:#F5FAFF;border-radius:18px;border:1.5px solid #DCE7F5">
        <img src="https://kidslearninglab.com/wp-content/uploads/2025/02/podcast-logo-app-rounded.png" style="width:48px;height:48px;border-radius:14px;display:block;margin:0 auto 20px">
        <h2 style="text-align:center;color:#14213D;margin-bottom:8px">Password change code</h2>
        <p style="text-align:center;color:#5B6B85;font-size:14px;line-height:1.6;margin-bottom:28px">Enter this code in Kids Learning Lab to confirm your password change. Expires in 10 minutes.</p>
        <div style="background:#fff;border:1.5px solid #DCE7F5;border-radius:14px;padding:28px;text-align:center;margin-bottom:24px">
          <span style="font-size:2.5rem;font-weight:900;letter-spacing:.25em;color:#1E6FE0">${cpCode}</span>
        </div>
        <p style="text-align:center;color:#5B6B85;font-size:12px">If you didn't request this, ignore this email.</p>
      </div>`;

    try {
      await fetch(RESEND_WORKER_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ to: auth.currentUser.email, subject: 'Your Kids Learning Lab password change code', html })
      });
      cpSendCodeBtn.style.display = 'none';
      cpVerifyCodeInput.style.display = 'block';
      cpVerifyCodeBtn.style.display = 'block';
      setTimeout(() => cpVerifyCodeInput.focus(), 80);
    } catch {
      cpShowError(2, 'Could not send code. Please try again.');
      cpSendCodeBtn.disabled = false;
      cpSendCodeBtn.textContent = 'Send Code to Email';
    }
  });

  cpVerifyCodeBtn.addEventListener('click', () => {
    const entered = cpVerifyCodeInput.value.trim();
    cpShowError(2, '');
    if (!entered || entered.length < 6) { cpShowError(2, 'Enter the 6-digit code.'); return; }
    if (!cpCode || Date.now() > cpCodeExpiry) { cpShowError(2, 'Code expired. Please resend.'); return; }
    if (entered !== cpCode) {
      cpShowError(2, 'Incorrect code. Try again.');
      cpVerifyCodeInput.value = '';
      return;
    }
    cpCode = null;
    closeCpModal(2);

    cpNewPassword.value = '';
    cpConfirmPassword.value = '';
    cpShowError(3, '');
    cpBtn3.disabled = false;
    cpBtn3.textContent = 'Set New Password';
    cpModal3.classList.add('show');
    setTimeout(() => cpNewPassword.focus(), 80);
  });

  // Step 3 — set the new password
  cpBtn3.addEventListener('click', async () => {
    const newPw = cpNewPassword.value;
    const confPw = cpConfirmPassword.value;
    cpShowError(3, '');
    if (!newPw || newPw.length < 6) { cpShowError(3, 'Password must be at least 6 characters.'); return; }
    if (newPw !== confPw) { cpShowError(3, 'Passwords do not match.'); return; }

    cpBtn3.disabled = true;
    cpBtn3.textContent = 'Saving…';
    try {
      await updatePassword(auth.currentUser, newPw);
      closeCpModal(3);
      settingsStatus.textContent = 'Password changed successfully!';
      settingsStatus.className = 'settings-status success';
    } catch (err) {
      cpShowError(3, err.message || 'Could not update password. Please try again.');
      cpBtn3.disabled = false;
      cpBtn3.textContent = 'Set New Password';
    }
  });

  document.getElementById('cpCancel1')?.addEventListener('click', () => closeCpModal(1));
  document.getElementById('cpCancel2')?.addEventListener('click', () => closeCpModal(2));
  document.getElementById('cpCancel3')?.addEventListener('click', () => closeCpModal(3));
}

async function showVerifyModal(email) {
  verifyEmail = email;
  verifyCode = generateCode();
  verifyExpiry = Date.now() + 10 * 60 * 1000;
  document.getElementById('verifyModalSubtitle').textContent = `We sent a 6-digit code to ${email}. It expires in 10 minutes.`;
  verifyCodeInput.value = '';
  verifyError.textContent = '';
  verifyModalOverlay.classList.add('show');
  startVerifyTimer();
  try { await sendVerificationEmail(email, verifyCode); }
  catch { verifyError.textContent = 'Could not send email. Try resending.'; }
  setTimeout(() => verifyCodeInput.focus(), 80);
}

verifySubmitBtn.addEventListener('click', () => {
  const entered = verifyCodeInput.value.trim();
  if (!entered || entered.length < 6) { verifyError.textContent = 'Please enter the 6-digit code.'; return; }
  if (!verifyCode || Date.now() > verifyExpiry) { verifyError.textContent = 'Code has expired. Please resend.'; return; }
  if (entered !== verifyCode) {
    verifyError.textContent = 'Incorrect code. Please try again.';
    verifyCodeInput.value = ''; verifyCodeInput.focus();
    return;
  }
  clearInterval(verifyTimerInterval);
  verifyCode = null;
  verifyModalOverlay.classList.remove('show');
  verificationInProgress = false;
  showUsernameModal();
});

verifyResendBtn.addEventListener('click', async () => {
  verifyResendBtn.disabled = true;
  verifyError.textContent = '';
  verifyCode = generateCode();
  verifyExpiry = Date.now() + 10 * 60 * 1000;
  startVerifyTimer();
  try {
    await sendVerificationEmail(verifyEmail, verifyCode);
    verifyError.style.color = 'var(--blue-main)';
    verifyError.textContent = 'New code sent!';
    setTimeout(() => { verifyError.textContent = ''; verifyError.style.color = 'var(--error)'; }, 3000);
  } catch {
    verifyError.textContent = 'Could not resend. Please try again.';
  } finally {
    setTimeout(() => verifyResendBtn.disabled = false, 8000);
  }
});

/* ═══════════════════════════════════════════════
   SECURITY: TOO-MANY-LOGINS CHECK (>25 sign-ins/hour)
   2 steps — confirm password, then an emailed code — reusing the
   same pattern as the Change Password flow above.
═══════════════════════════════════════════════ */
const secCheckModal1 = document.getElementById('secCheckModal1');
const secCheckModal2 = document.getElementById('secCheckModal2');
const secCheckPassword = document.getElementById('secCheckPassword');
const secCheckError1 = document.getElementById('secCheckError1');
const secCheckBtn1 = document.getElementById('secCheckBtn1');
const secCheckCancel1 = document.getElementById('secCheckCancel1');
const secCheckEmailDisplay = document.getElementById('secCheckEmailDisplay');
const secCheckSendCodeBtn = document.getElementById('secCheckSendCodeBtn');
const secCheckVerifyCodeBtn = document.getElementById('secCheckVerifyCodeBtn');
const secCheckCodeInput = document.getElementById('secCheckCodeInput');
const secCheckError2 = document.getElementById('secCheckError2');
const secCheckCancel2 = document.getElementById('secCheckCancel2');

function runSecurityCheck(user) {
  return new Promise((resolve) => {
    if (!secCheckModal1 || !secCheckModal2) { resolve(true); return; }

    let secCode = null;
    let secCodeExpiry = null;
    let resolved = false;

    function cleanupListeners() {
      secCheckBtn1.removeEventListener('click', onBtn1);
      secCheckCancel1.removeEventListener('click', onCancel);
      secCheckSendCodeBtn.removeEventListener('click', onSendCode);
      secCheckVerifyCodeBtn.removeEventListener('click', onVerifyCode);
      secCheckCancel2.removeEventListener('click', onCancel);
    }

    function finish(ok) {
      if (resolved) return;
      resolved = true;
      secCheckModal1.classList.remove('show');
      secCheckModal2.classList.remove('show');
      cleanupListeners();
      resolve(ok);
    }

    async function onBtn1() {
      const pw = secCheckPassword.value;
      secCheckError1.textContent = '';
      if (!pw) { secCheckError1.textContent = 'Please enter your password.'; return; }
      secCheckBtn1.disabled = true;
      secCheckBtn1.textContent = 'Checking…';
      try {
        await signInWithEmailAndPassword(auth, user.email, pw);
        secCheckModal1.classList.remove('show');
        secCheckEmailDisplay.textContent = user.email || '';
        secCheckCodeInput.value = '';
        secCheckCodeInput.style.display = 'none';
        secCheckVerifyCodeBtn.style.display = 'none';
        secCheckSendCodeBtn.style.display = 'block';
        secCheckSendCodeBtn.disabled = false;
        secCheckSendCodeBtn.textContent = 'Send Code to Email';
        secCheckError2.textContent = '';
        secCheckModal2.classList.add('show');
      } catch {
        secCheckError1.textContent = 'Incorrect password. Please try again.';
      } finally {
        secCheckBtn1.disabled = false;
        secCheckBtn1.textContent = 'Confirm Password';
      }
    }

    async function onSendCode() {
      secCheckSendCodeBtn.disabled = true;
      secCheckSendCodeBtn.textContent = 'Sending…';
      secCode = generateCode();
      secCodeExpiry = Date.now() + 10 * 60 * 1000;
      const html = `
        <div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:40px 32px;background:#F5FAFF;border-radius:18px;border:1.5px solid #DCE7F5">
          <img src="https://kidslearninglab.com/wp-content/uploads/2025/02/podcast-logo-app-rounded.png" style="width:48px;height:48px;border-radius:14px;display:block;margin:0 auto 20px">
          <h2 style="text-align:center;color:#14213D;margin-bottom:8px">Security check code</h2>
          <p style="text-align:center;color:#5B6B85;font-size:14px;line-height:1.6;margin-bottom:28px">We noticed a lot of sign-ins to your account. Enter this code to confirm it's really you. It expires in 10 minutes.</p>
          <div style="background:#fff;border:1.5px solid #DCE7F5;border-radius:14px;padding:28px;text-align:center;margin-bottom:24px">
            <span style="font-size:2.5rem;font-weight:900;letter-spacing:.25em;color:#1E6FE0">${secCode}</span>
          </div>
          <p style="text-align:center;color:#5B6B85;font-size:12px">If you didn't request this, ignore this email.</p>
        </div>`;
      try {
        await fetch(RESEND_WORKER_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ to: user.email, subject: 'Your Kids Learning Lab security code', html })
        });
        secCheckSendCodeBtn.style.display = 'none';
        secCheckCodeInput.style.display = 'block';
        secCheckVerifyCodeBtn.style.display = 'block';
        setTimeout(() => secCheckCodeInput.focus(), 80);
      } catch {
        secCheckError2.textContent = 'Could not send code. Please try again.';
        secCheckSendCodeBtn.disabled = false;
        secCheckSendCodeBtn.textContent = 'Send Code to Email';
      }
    }

    function onVerifyCode() {
      const entered = secCheckCodeInput.value.trim();
      secCheckError2.textContent = '';
      if (!entered || entered.length < 6) { secCheckError2.textContent = 'Enter the 6-digit code.'; return; }
      if (!secCode || Date.now() > secCodeExpiry) { secCheckError2.textContent = 'Code expired. Please resend.'; return; }
      if (entered !== secCode) {
        secCheckError2.textContent = 'Incorrect code. Try again.';
        secCheckCodeInput.value = '';
        return;
      }
      finish(true);
    }

    function onCancel() { finish(false); }

    secCheckPassword.value = '';
    secCheckError1.textContent = '';
    secCheckBtn1.disabled = false;
    secCheckBtn1.textContent = 'Confirm Password';
    secCheckModal1.classList.add('show');
    setTimeout(() => secCheckPassword.focus(), 80);

    secCheckBtn1.addEventListener('click', onBtn1);
    secCheckCancel1.addEventListener('click', onCancel);
    secCheckSendCodeBtn.addEventListener('click', onSendCode);
    secCheckVerifyCodeBtn.addEventListener('click', onVerifyCode);
    secCheckCancel2.addEventListener('click', onCancel);
  });
}

// Runs after a successful password sign-in (not signup): checks the
// login-rate limit before letting the app reveal itself, then records
// this login.
async function handlePostSignInChecks() {
  const user = auth.currentUser;
  if (!user) { verificationInProgress = false; return; }

  const attemptCount = recordLoginAttempt();
  if (attemptCount > LOGIN_RATE_LIMIT) {
    const ok = await runSecurityCheck(user);
    if (!ok) {
      verificationInProgress = false;
      await logout();
      showMessage('Sign-in cancelled.', 'error');
      return;
    }
  }

  try {
    await setDoc(doc(db, 'userProfiles', user.uid), { lastLoginAt: Date.now() }, { merge: true });
  } catch {
    // non-fatal — worst case the stale-login check re-triggers next time
  }

  verificationInProgress = false;
  await proceedAfterAuth(user);
}

function showUsernameModal(isChange = false) {
  usernameInput.value = isChange ? (auth.currentUser?.displayName || '') : '';
  usernameError.textContent = '';
  usernameModalOverlay.classList.add('show');
  usernameModalOverlay.dataset.mode = isChange ? 'change' : 'onboarding';
  document.getElementById('usernameModalTitle').textContent = isChange ? 'Change Your Name' : 'Choose your username';
  usernameSubmitBtn.textContent = isChange ? 'Save Name' : 'Set Username';
  setTimeout(() => usernameInput.focus(), 80);
}

profileChangeNameBtn.addEventListener('click', () => showUsernameModal(true));

usernameSubmitBtn.addEventListener('click', async () => {
  const val = usernameInput.value.trim();
  const isChange = usernameModalOverlay.dataset.mode === 'change';
  if (!val || val.length < 2) { usernameError.textContent = 'Username must be at least 2 characters.'; return; }
  if (!USERNAME_ALLOWED.test(val)) { usernameError.textContent = 'Only letters, numbers, and spaces allowed.'; return; }
  if (RESERVED_NAME_PATTERN.test(val)) { usernameError.textContent = "That name isn't available."; return; }
  usernameSubmitBtn.disabled = true;
  usernameSubmitBtn.textContent = isChange ? 'Saving…' : 'Saving…';
  try {
    await updateProfile(auth.currentUser, { displayName: val });
    const profileUpdate = { usernameSet: true, displayName: val };
    if (!isChange) profileUpdate.lastLoginAt = Date.now(); // first-ever login for a brand-new account
    await setDoc(doc(db, 'userProfiles', auth.currentUser.uid), profileUpdate, { merge: true });
    usernameModalOverlay.classList.remove('show');
    if (!isChange) {
      authScreen.style.display = 'none';
      appShell.style.display = 'flex';
    }
    updateSettingsUI(auth.currentUser);
  } catch (err) {
    console.error('Username save failed:', err.code, err.message);
    usernameError.textContent = 'Could not save username. Try again.';
  } finally {
    usernameSubmitBtn.disabled = false;
    usernameSubmitBtn.textContent = isChange ? 'Save Name' : 'Set Username';
  }
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  clearMessage();
  const email = emailInput.value.trim();
  const password = passwordInput.value;

  submitBtn.disabled = true;
  submitBtn.textContent = mode === 'signin' ? 'Signing In…' : 'Creating Account…';

  if (mode === 'signup') {
    verificationInProgress = true;
    // Show the "Creating your account…" full-pager right away — since it's
    // a first-time overlay (see firstTimeOverlays.js), everyone only ever
    // sees it once, but that once is always their real first sign-up, never
    // a pre-existing account. The auto-advance timer runs in parallel with
    // the actual signUp() call rather than blocking on it, so it never
    // makes account creation feel slower than it already is.
    const overlayPromise = maybeShowOverlay('accountCreating', { vars: { email } });
    const result = await signUp(email, password);
    await overlayPromise;
    submitBtn.disabled = false;
    submitBtn.textContent = 'Create Account';
    if (!result.success) {
      verificationInProgress = false;
      showMessage(result.error, 'error');
      return;
    }
    await showVerifyModal(email);
  } else {
    verificationInProgress = true; // hold the app reveal until our security checks below pass
    const result = await signIn(email, password);
    submitBtn.disabled = false;
    submitBtn.textContent = 'Sign In';
    if (!result.success) {
      verificationInProgress = false;
      showMessage(result.error, 'error');
      return;
    }
    await handlePostSignInChecks();
  }
});

// ---- Auth state -> screen swap ----
let lastAuthUid = null; // tracks whose data is currently loaded, so we can
                         // clear stale in-memory state when the account changes

// Reveals the app shell for an already-authenticated user. Called both from
// onAuthStateChanged (e.g. an already-signed-in user reopening the app)
// and directly from handlePostSignInChecks after a fresh sign-in clears
// its security checks.
async function proceedAfterAuth(user) {
  try {
    const profileSnap = await getDoc(doc(db, 'userProfiles', user.uid));
    if (!profileSnap.exists() || !profileSnap.data()?.usernameSet) {
      authScreen.style.display = 'none';
      appShell.style.display = 'none';
      showUsernameModal();
      return;
    }
  } catch {
    // fail open
  }

  authScreen.style.display = 'none';
  appShell.style.display = 'flex';
  if (shouldPlayStartupVideo()) playStartupVideo(); // overlay sits above appShell, fades out into it
  updateSettingsUI(user);   // ← make sure this line is here
  await initBadgesForUser(user.uid);
  await initFirstTimeOverlaysForUser(user.uid);
  await initTutorialForUser(user.uid);
  checkAccountBadge(true); // usernameSet (checked above) implies email verification + username both done
  startXpListener(user);
  startNotifBell(user);
  initPushForCurrentUser(); // ask for push permission on first app open, not just on first bell tap
  initPremiumForCurrentUser(); // configures RevenueCat + starts the Firestore premium listener
  initShopStateForCurrentUser(); // starts the shared XP Shop fields listener (unlocked colors/emoji, streak passes, etc)
  startHomeMirrorForCurrentUser(); // starts the Firestore→RTDB background mirror loadHomeData() reads from
  refreshHome();
  maybeRunAppTour(); // no-op if already seen; needs refreshHome()'s stat row/shop banner to already be visible
}

async function handleAuthStateChange(user) {
  if (user?.uid !== lastAuthUid) {
    // Different account (or signed out) — wipe any cached data from the
    // previous account before loading/showing the new one. This includes
    // any full-page overlay/modal that might still be marked "show" (e.g.
    // Account Settings, opened from within itself via Sign Out) and the
    // in-memory profile photo, which otherwise keeps showing the previous
    // account's picture until the app is fully restarted.
    resetLearnState();
    resetBadgesState();
    resetFirstTimeOverlaysState();
    resetTutorialState();
    resetPremiumState();
    resetShopState();
    stopHomeMirror();
    stopXpListener();
    stopNotifBell();
    document.querySelectorAll('.kll-modal-overlay.show').forEach((el) => el.classList.remove('show'));
    currentAvatar = null;
    renderAvatarInto(settingsAvatar, null);
    welcomeBackCheckedThisSession = false;
    lastAuthUid = user?.uid || null;
  }

  if (user) {
    if (verificationInProgress || onboardingInProgress) return; // a security-check or onboarding flow is in progress — it will reveal the app itself when ready
    await proceedAfterAuth(user);
  } else {
    if (onboardingInProgress) return; // wizard owns the screen for a signed-out first-time device — don't show the plain auth screen underneath it
    authScreen.style.display = 'flex';
    appShell.style.display = 'none';
    loadLastUsedAccount();
  }
}
onAuthStateChanged(auth, handleAuthStateChange);

// ---- First-run onboarding wizard: DISCONNECTED ----
// onboarding.js is no longer called from here. onboardingInProgress stays
// permanently false, so every code path that checks it (the
// onAuthStateChanged callback above, proceedAfterAuth) behaves exactly as
// if onboarding never existed — a signed-out device just sees the normal
// auth-screen sign-in/sign-up form, same as before onboarding.js was ever
// wired in. onboarding.js itself and its markup in index.html are left in
// place untouched (nothing to break by leaving them), just no longer
// reachable from anywhere.

// ---- Bottom nav ----
const navButtons = document.querySelectorAll('.nav-btn');
const pages = document.querySelectorAll('.page');

// The bell lives outside the page divs (fixed-position over the app shell),
// so it doesn't get hidden/shown by the page-switching logic below on its
// own — it only makes sense on Listen (episode notifications) and Profile
// (friend requests/accepts), so toggle it explicitly alongside the page.
const NOTIF_BELL_PAGES = new Set(['listen', 'settings']);
function updateNotifBellVisibility(target) {
  const notifBellBtnEl = document.getElementById('notifBellBtn');
  if (notifBellBtnEl) notifBellBtnEl.style.display = NOTIF_BELL_PAGES.has(target) ? 'flex' : 'none';
}

// Left-to-right order the pages appear in the navbar. A page to the right
// of the current one (higher index) slides in from the right as it becomes
// active, and vice versa — matching how swiping between tabs usually feels.
const PAGE_ORDER = ['listen', 'learn', 'home', 'review', 'settings'];
const PAGE_TRANSITION_MS = 340; // matches the .page transform transition duration in CSS
let isPageTransitioning = false;

function switchPage(target) {
  const current = document.querySelector('.nav-btn.active')?.dataset.page;
  if (!current || current === target || isPageTransitioning) return;

  navButtons.forEach((b) => b.classList.toggle('active', b.dataset.page === target));

  const oldPageEl = document.getElementById(`page-${current}`);
  const newPageEl = document.getElementById(`page-${target}`);

  // Fallback to the old instant swap if either page element is missing —
  // keeps navigation working even if a page id ever gets renamed.
  if (!oldPageEl || !newPageEl) {
    pages.forEach((p) => {
      p.style.display = p.id === `page-${target}` ? 'flex' : 'none';
    });
    updateNotifBellVisibility(target);
    if (target === 'home') refreshHome();
    if (target === 'review') window.renderReviewPage?.();
    return;
  }

  const oldIdx = PAGE_ORDER.indexOf(current);
  const newIdx = PAGE_ORDER.indexOf(target);
  const enterFromRight = newIdx > oldIdx; // target sits to the right of current in the navbar

  isPageTransitioning = true;

  // Place the incoming page off-screen on the correct side with transitions
  // disabled, so the very first frame doesn't animate from wherever it was
  // last left sitting.
  newPageEl.style.transition = 'none';
  newPageEl.style.display = 'flex';
  newPageEl.style.transform = `translateX(${enterFromRight ? '100%' : '-100%'})`;
  newPageEl.style.opacity = '0';

  // Force a reflow so the browser commits that starting position before we
  // flip both pages toward their animated end states below.
  void newPageEl.offsetWidth;
  newPageEl.style.transition = '';

  oldPageEl.style.transform = `translateX(${enterFromRight ? '-100%' : '100%'})`;
  oldPageEl.style.opacity = '0';
  newPageEl.style.transform = 'translateX(0)';
  newPageEl.style.opacity = '1';

  setTimeout(() => {
    oldPageEl.style.display = 'none';
    oldPageEl.style.transition = 'none';
    oldPageEl.style.transform = 'translateX(0)';
    oldPageEl.style.opacity = '1';
    void oldPageEl.offsetWidth;
    oldPageEl.style.transition = '';
    isPageTransitioning = false;
  }, PAGE_TRANSITION_MS);

  updateNotifBellVisibility(target);
  if (target === 'home') refreshHome();
  if (target === 'review') window.renderReviewPage?.();
}

navButtons.forEach((btn) => {
  btn.addEventListener('click', () => switchPage(btn.dataset.page));
});

// Set correct initial visibility for whichever nav button is active on
// first load (currently "home", where the bell should be hidden).
updateNotifBellVisibility(document.querySelector('.nav-btn.active')?.dataset.page);

// ============================================================
// LAST USED ACCOUNT — quick sign-in box shown above the email field on
// the auth screen, populated from whichever account most recently signed
// out. Stores only the email (never a password) in localStorage.
// ============================================================
const LAST_USED_ACCOUNT_KEY = 'kll_last_used_account';
const lastUsedAccountCard = document.getElementById('lastUsedAccountCard');
const lastUsedAccountEmail = document.getElementById('lastUsedAccountEmail');
const quickSigninOverlay = document.getElementById('quickSigninModalOverlay');
const quickSigninEmailEl = document.getElementById('quickSigninEmail');
const quickSigninPasswordInput = document.getElementById('quickSigninPasswordInput');
const quickSigninError = document.getElementById('quickSigninError');
const quickSigninSubmitBtn = document.getElementById('quickSigninSubmitBtn');
const quickSigninSwitchBtn = document.getElementById('quickSigninSwitchBtn');

function saveLastUsedAccount() {
  const email = auth.currentUser?.email;
  if (!email) return;
  try {
    localStorage.setItem(LAST_USED_ACCOUNT_KEY, JSON.stringify({ email }));
  } catch { /* localStorage unavailable — quick sign-in box just won't show next time */ }
}

function clearLastUsedAccount() {
  try { localStorage.removeItem(LAST_USED_ACCOUNT_KEY); } catch {}
  lastUsedAccountCard?.classList.remove('show');
}

function loadLastUsedAccount() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(LAST_USED_ACCOUNT_KEY) || 'null'); } catch {}
  if (!saved || !saved.email || !lastUsedAccountCard) { lastUsedAccountCard?.classList.remove('show'); return; }

  lastUsedAccountEmail.textContent = saved.email;
  lastUsedAccountCard.classList.add('show');
}

lastUsedAccountCard?.addEventListener('click', () => {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(LAST_USED_ACCOUNT_KEY) || 'null'); } catch {}
  if (!saved || !saved.email) return;

  quickSigninEmailEl.textContent = saved.email;
  quickSigninPasswordInput.value = '';
  quickSigninError.textContent = '';
  quickSigninOverlay._email = saved.email;
  quickSigninOverlay.classList.add('show');
  setTimeout(() => quickSigninPasswordInput.focus(), 300);
});

quickSigninSwitchBtn?.addEventListener('click', () => {
  quickSigninOverlay.classList.remove('show');
  clearLastUsedAccount();
});

async function submitQuickSignin() {
  const email = quickSigninOverlay._email;
  const pw = quickSigninPasswordInput.value;
  if (!email || !pw) {
    quickSigninError.textContent = 'Enter your password.';
    return;
  }
  quickSigninSubmitBtn.disabled = true;
  quickSigninSubmitBtn.textContent = 'Signing In…';
  quickSigninError.textContent = '';
  try {
    await signInWithEmailAndPassword(auth, email, pw);
    quickSigninOverlay.classList.remove('show');
    // onAuthStateChanged handles the screen swap into the app shell
  } catch (err) {
    quickSigninError.textContent = err.code === 'auth/wrong-password' || err.code === 'auth/invalid-credential'
      ? 'Incorrect password.'
      : (err.message || 'Could not sign in.');
  } finally {
    quickSigninSubmitBtn.disabled = false;
    quickSigninSubmitBtn.textContent = 'Sign In';
  }
}

quickSigninSubmitBtn?.addEventListener('click', submitQuickSignin);
quickSigninPasswordInput?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') submitQuickSignin();
});

loadLastUsedAccount();
// ============================================================
// HOME PAGE 2.0 — fixed-role sections (hero / stats / notifications /
// games) instead of a randomized packed widget grid. Each section owns
// its own layout strategy for "however many items exist today", so no
// combination of optional items can leave a gap the way the old dense
// grid could.
// ============================================================
const homeGreeting = document.getElementById('homeGreeting');
const homeLoadingEl = document.getElementById('homeLoading');
const homeSkeletonEl = document.getElementById('homeSkeleton');
const homeHeroEl = document.getElementById('homeHero');
const homeHeroLabelEl = document.getElementById('homeHeroLabel');
const homeHeroTitleEl = document.getElementById('homeHeroTitle');
const homeHeroSubEl = document.getElementById('homeHeroSub');
const homeHeroProgressTrackEl = document.getElementById('homeHeroProgressTrack');
const homeHeroProgressFillEl = document.getElementById('homeHeroProgressFill');
const homeUsageBannerEl = document.getElementById('homeUsageBanner');
const homeStatsRowEl = document.getElementById('homeStatsRow');
const homeStatStreakEl = document.getElementById('homeStatStreak');
const homeStatStreakNumEl = document.getElementById('homeStatStreakNum');
const homeStatStreakLabelEl = document.getElementById('homeStatStreakLabel');
const homeStatXpNumEl = document.getElementById('homeStatXpNum');
const homeStatXpEl = document.getElementById('homeStatXp');
const homeStatFriendsEl = document.getElementById('homeStatFriends');
const homeStatFriendsNumEl = document.getElementById('homeStatFriendsNum');
const homeStatReviewEl = document.getElementById('homeStatReview');
const homeStatReviewNumEl = document.getElementById('homeStatReviewNum');
const homeStatReviewLabelEl = document.getElementById('homeStatReviewLabel');
const homeXpShopBannerEl = document.getElementById('homeXpShopBanner');
const homeNotifSectionEl = document.getElementById('homeNotifSection');
const homeNotifListEl = document.getElementById('homeNotifList');
const homeGamesSectionEl = document.getElementById('homeGamesSection');
const homeGamesRowEl = document.getElementById('homeGamesRow');
const homeEditProfileBtn = document.getElementById('homeEditProfileBtn');
const homeReviewPageBtn = document.getElementById('homeReviewPageBtn');

homeEditProfileBtn?.addEventListener('click', () => document.querySelector('.nav-btn[data-page="settings"]')?.click());
// Review Page itself is free to open — the Premium gate lives inside it,
// on the individual Full Weak Spot Review / Full Personalized Review / Daily
// Lesson actions (openReviewPage() no-ops if Review Lessons are toggled off).
homeReviewPageBtn?.addEventListener('click', async () => { await ensureLearnInitialized(); openReviewPage(); });
homeStatXpEl?.addEventListener('click', () => openXpShop());
homeXpShopBannerEl?.addEventListener('click', () => openXpShop());

let homeLoadToken = 0; // bumped on every refresh so a slow, stale fetch can't clobber a newer render

// Every playable game (plus Join Game), with the exact card/button ID to
// trigger from the Learning Games page so a tap can jump straight into one
// instead of opening the games list first. All 9 always show, in the same
// order as the Learning Games page.
const HOME_GAME_LIST = [
  { name: 'Join Game', desc: "Enter a friend's code to play together", cardId: 'joinGameEntryBtn', color: 'blue' },
  { name: 'Trivia', desc: 'Quick-fire questions about anything', cardId: 'gameCardTrivia', color: 'blue' },
  { name: 'Voice Trivia', desc: 'Say your answer out loud — the app listens', cardId: 'gameCardVoiceTrivia', color: 'cyan' },
  { name: 'Maze', desc: 'Navigate a maze, answer questions to keep moving', cardId: 'gameCardMaze', color: 'teal' },
  { name: 'Seesaw', desc: '2 players, pass the phone — answer before time runs out', cardId: 'gameCardSeesaw', color: 'amber' },
  { name: 'Meltdown', desc: 'Solo speed round — answer fast before you melt', cardId: 'gameCardMeltdown', color: 'purple' },
  { name: 'Word Grid', desc: 'Guess the secret 5-letter word in 6 tries', cardId: 'gameCardWordGrid', color: 'green' },
  { name: 'Who Can Answer First?', desc: '2 players, same question — fastest correct tap wins', cardId: 'gameCardDuel', color: 'pink' },
  { name: 'Word Connectors', desc: 'Find groups of 4 across all your courses', cardId: 'gameCardConnectors', color: 'rose' },
];

function todayStrHome() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

// ---- Home-data cache (instant paint on login) ----
// Purely a "show something immediately" cache, not a source of truth —
// every refreshHome() still runs the real loadHomeData() fetch and
// overwrites both the render and this cache with fresh results right
// after. Keyed per-uid so switching accounts on one device can't flash
// the previous account's Home data.
function homeDataCacheKey(uidStr) {
  return `kll_home_data_${uidStr}`;
}
function getCachedHomeData(uidStr) {
  try {
    const raw = localStorage.getItem(homeDataCacheKey(uidStr));
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}
function setCachedHomeData(uidStr, data) {
  try {
    localStorage.setItem(homeDataCacheKey(uidStr), JSON.stringify(data));
  } catch (err) {
    console.warn('Could not cache Home data:', err);
  }
}

// Jumps to the Learn tab exactly like tapping the nav button would (page
// swap + Learn's own lazy init), so widgets can send the learner there.
function goToLearnTab() {
  document.querySelector('.nav-btn[data-page="learn"]')?.click();
}

async function refreshHome() {
  const u = auth.currentUser;
  if (!u || !homeHeroEl) return;
  const token = ++homeLoadToken;

  const name = u.displayName || (u.email ? u.email.split('@')[0] : 'Learner');
  if (homeGreeting) homeGreeting.textContent = `Welcome, ${name}!`;

  // Paint immediately from whatever Home data was last cached for this
  // account, so the screen isn't blank/skeleton on every login while the
  // real Firestore reads in loadHomeData() are still in flight. This is
  // deliberately just a first paint, not a replacement for the real fetch
  // below — it can be a session or more stale (course progress, streak,
  // XP, friend requests may have changed since), so the real data below
  // still always runs and overwrites it once it resolves.
  const cachedHome = getCachedHomeData(u.uid);
  if (cachedHome) renderHome(cachedHome);

  const data = await loadHomeData(u.uid);
  if (token !== homeLoadToken) return; // a newer refresh already started
  renderHome(data);
  setCachedHomeData(u.uid, data);
  maybeShowWelcomeBackOverlay(u);
  maybeShowWeeklyReviewOverlay(u);
}

document.addEventListener('kll:refreshHome', () => refreshHome());

// ---- Welcome Back full-pager (20+ days since last app open) ----
// Reads/writes the same lastLoginAt written in handlePostSignInChecks on
// every sign-in, so this needs no new Firestore field — "20+ days since
// you last opened the app" is exactly what that timestamp already tracks.
// Runs once per Home refresh cycle at most (see the module-scoped guard),
// since refreshHome() can fire more than once in a session.
let welcomeBackCheckedThisSession = false;
async function maybeShowWelcomeBackOverlay(u) {
  if (welcomeBackCheckedThisSession) return;
  welcomeBackCheckedThisSession = true;
  try {
    const snap = await getDoc(doc(db, 'userProfiles', u.uid));
    const lastLoginAt = snap.exists() ? snap.data()?.lastLoginAt : null;
    if (!lastLoginAt) return;
    const days = Math.floor((Date.now() - lastLoginAt) / 86400000);
    if (days >= 20) {
      await maybeShowOverlay('welcomeBack', { vars: { days: String(days) } });
    }
  } catch (err) {
    console.error('Welcome Back overlay check failed:', err);
  }
}

// ---- Weekly Review full-pager (Premium only) ----
// Shows once per calendar week, the first time the app is opened that
// week — same "session guard + userProfiles doc" shape as
// maybeShowWelcomeBackOverlay above, just keyed off a week number instead
// of a day gap. Deliberately its own lightweight overlay (see
// #weeklyReviewOverlay in index.html) rather than routed through
// firstTimeOverlays.js, since this isn't a one-time "first time you saw
// this feature" beat — it's meant to recur every week, forever.
let weeklyReviewCheckedThisSession = false;
function isoWeekKey(d) {
  // Sunday-based week bucket ("YYYY-Wnn" of the days-since-epoch // 7) —
  // doesn't need to match the ISO calendar standard exactly, just needs to
  // change once a week and be cheap to compute/compare.
  const days = Math.floor(d.getTime() / 86400000);
  return String(Math.floor(days / 7));
}
async function maybeShowWeeklyReviewOverlay(u) {
  if (weeklyReviewCheckedThisSession) return;
  weeklyReviewCheckedThisSession = true;
  if (!isPremium()) return; // Premium-only feature — free users never see this
  try {
    const profileRef = doc(db, 'userProfiles', u.uid);
    const snap = await getDoc(profileRef);
    const lastShownWeek = snap.exists() ? snap.data()?.lastWeeklyReviewWeek : null;
    const thisWeek = isoWeekKey(new Date());
    if (lastShownWeek === thisWeek) return; // already seen this week

    const data = await getWeeklyReviewData();
    renderWeeklyReviewOverlay(data);
    await setDoc(profileRef, { lastWeeklyReviewWeek: thisWeek, lastWeeklyReview: data }, { merge: true });
  } catch (err) {
    console.error('Weekly Review overlay check failed:', err);
  }
}

const weeklyReviewOverlay = document.getElementById('weeklyReviewOverlay');
const weeklyReviewRecapEl = document.getElementById('weeklyReviewRecap');
const weeklyReviewXpEl = document.getElementById('weeklyReviewXp');
const weeklyReviewStreakEl = document.getElementById('weeklyReviewStreak');
const weeklyReviewCoursesEl = document.getElementById('weeklyReviewCourses');
const weeklyReviewToughestEl = document.getElementById('weeklyReviewToughest');
const weeklyReviewCloseBtn = document.getElementById('weeklyReviewCloseBtn');
const weeklyReviewComboBtn = document.getElementById('weeklyReviewComboBtn');

function renderWeeklyReviewOverlay(data) {
  if (!weeklyReviewOverlay) {
    console.warn('Weekly Review markup missing from index.html (#weeklyReviewOverlay).');
    return;
  }
  if (weeklyReviewRecapEl) {
    // AI-generated recap sentence when the textonlygroqfast call succeeded;
    // otherwise a plain templated fallback so the overlay never shows blank.
    weeklyReviewRecapEl.textContent = data.recap || (data.totalMistakes > 0
      ? `Great effort this week! You've earned ${data.xp} XP across ${data.courseCount} course${data.courseCount === 1 ? '' : 's'} — keep it up!`
      : `Great effort this week! You've earned ${data.xp} XP and you're all caught up — keep it up!`);
  }
  if (weeklyReviewXpEl) weeklyReviewXpEl.textContent = String(data.xp);
  if (weeklyReviewStreakEl) weeklyReviewStreakEl.textContent = String(data.streak);
  if (weeklyReviewCoursesEl) weeklyReviewCoursesEl.textContent = String(data.courseCount);
  if (weeklyReviewToughestEl) {
    weeklyReviewToughestEl.textContent = data.toughestCourse
      ? `Toughest course: ${data.toughestCourse.title} (${data.toughestCourse.count} to review)`
      : "You're all caught up — no mistakes waiting anywhere!";
  }
  // Only worth pointing at Combo Lesson if there's actually more than one
  // course AND something to review — otherwise it'd just paywall-loop into
  // "nothing to combo yet."
  if (weeklyReviewComboBtn) {
    weeklyReviewComboBtn.style.display = (data.courseCount >= 2 && data.totalMistakes > 0) ? '' : 'none';
  }
  weeklyReviewOverlay.classList.add('show');
}
weeklyReviewCloseBtn?.addEventListener('click', () => weeklyReviewOverlay.classList.remove('show'));
weeklyReviewComboBtn?.addEventListener('click', async () => {
  weeklyReviewOverlay.classList.remove('show');
  await ensureLearnInitialized();
  goToLearnTab();
  openComboLesson();
});

// ---- Settings: "See Last Week's Recap" button ----
// Premium-only, same as the auto overlay. Two cases:
//   - Account is 7+ days old and already has a stored recap → just show
//     that exact stored recap again, no regeneration.
//   - Brand-new account (under 7 days old) with nothing stored yet →
//     generate one now for however many days they've actually been
//     around, then store it AS THIS WEEK'S recap (so it also satisfies
//     the normal once-a-week auto-overlay and doesn't generate twice).
const settingsWeeklyRecapBtn = document.getElementById('settingsWeeklyRecapBtn');
settingsWeeklyRecapBtn?.addEventListener('click', async () => {
  const u = auth.currentUser;
  if (!u) return;
  if (!isPremium()) {
    openPaywall({ reason: 'Weekly Review is a Premium feature — a short AI recap of your progress.' });
    return;
  }
  try {
    const profileRef = doc(db, 'userProfiles', u.uid);
    const snap = await getDoc(profileRef);
    const profileData = snap.exists() ? snap.data() : {};

    if (profileData.lastWeeklyReview) {
      renderWeeklyReviewOverlay(profileData.lastWeeklyReview);
      return;
    }

    // Nothing generated yet — figure out how many days old the account is
    // from Firebase Auth's own creation timestamp (no app-side signup date
    // is tracked separately, so this is the one source of truth for it).
    const createdAtMs = u.metadata?.creationTime ? new Date(u.metadata.creationTime).getTime() : Date.now();
    const daysActive = Math.max(1, Math.floor((Date.now() - createdAtMs) / 86400000));

    await ensureLearnInitialized();
    const data = await getWeeklyReviewData(daysActive < 7 ? daysActive : undefined);
    renderWeeklyReviewOverlay(data);
    // Stored "as if it were 7 days ago" per spec — i.e. as this week's
    // recap, so the normal auto-overlay logic (isoWeekKey comparison)
    // treats it as already satisfied and won't immediately fire again on
    // the very next Home refresh.
    await setDoc(profileRef, { lastWeeklyReviewWeek: isoWeekKey(new Date()), lastWeeklyReview: data }, { merge: true });
  } catch (err) {
    console.error('See Last Week\'s Recap failed:', err);
  }
});


// (see homeMirror.js) instead of hitting Firestore directly — the mirror's
// listeners keep homeData/{uid}/... current in the background for as long
// as the user is signed in, so this is normally a single fast RTDB read
// instead of 4 separate Firestore reads/queries. Falls back to the direct
// Firestore reads below only if the mirror path is empty (e.g. the very
// first Home load right after sign-in, before the mirror's listeners have
// delivered their first snapshot yet) — Firestore itself never stops being
// the source of truth, this is purely which one loadHomeData() reads from.
async function loadHomeData(uidStr) {
  await ensureLearnInitialized(); // see comment on the old Promise.all below — same reason, unchanged

  let mirrored = null;
  try {
    const snap = await rtdbGet(rtdbRef(rtdb, `homeData/${uidStr}`));
    mirrored = snap.exists() ? snap.val() : null;
  } catch (err) {
    console.warn('loadHomeData: RTDB mirror read failed, falling back to Firestore:', err);
  }

  const learnProfile = mirrored?.learnProfile || { xp: 0 };
  const activeCourse = mirrored?.activeCourse !== undefined ? mirrored.activeCourse : await loadActiveCourseFromFirestore(uidStr);
  const wrongCount = mirrored?.wrongCount !== undefined ? mirrored.wrongCount
    : (activeCourse ? await loadWrongCountFromFirestore(uidStr, activeCourse.id) : 0);
  const friendReqCount = mirrored?.friendReqCount !== undefined ? mirrored.friendReqCount : 0;
  const friendsCount = mirrored?.friendsCount !== undefined ? mirrored.friendsCount : 0;
  const sharedCourseCount = mirrored?.sharedCourseCount !== undefined ? mirrored.sharedCourseCount : 0;

  // Mirror wasn't populated at all yet (fresh sign-in, listeners haven't
  // delivered a first snapshot) — do the original direct Firestore reads
  // for friends/sharedCourses too, rather than silently showing 0s.
  let finalFriendReqCount = friendReqCount, finalFriendsCount = friendsCount, finalSharedCourseCount = sharedCourseCount;
  if (!mirrored) {
    const [friendsSnap, sharedSnap] = await Promise.all([
      getDocs(collection(db, 'users', uidStr, 'friends')).catch(() => null),
      getDocs(query(collection(db, 'users', uidStr, 'sharedCourses'), where('status', '==', 'pending'))).catch(() => null),
    ]);
    finalFriendReqCount = 0; finalFriendsCount = 0;
    if (friendsSnap) {
      friendsSnap.forEach((d) => {
        const fd = d.data();
        if (fd.status === 'pending' && fd.direction === 'received') finalFriendReqCount++;
        if (fd.status === 'accepted') finalFriendsCount++;
      });
    }
    finalSharedCourseCount = sharedSnap ? sharedSnap.size : 0;
  }

  // Read straight from learn.js's own in-memory state — the exact same
  // numbers driving Learn's top-left streak counter — instead of
  // recomputing a second, cruder decay here that could disagree with it
  // (e.g. this used to ignore Streak Passes entirely). Unchanged from
  // before — streak is computed, not fetched, so it was never part of the
  // Firestore-vs-RTDB question.
  const { streak: effectiveStreak, missedDaysInRow, lastLessonDate } = getLearnStreakSnapshot();
  const doneToday = lastLessonDate === todayStrHome();

  const xp = learnProfile.xp || 0;

  // Latest episode: read straight off the Listen tab's own DOM once it's
  // rendered its list — avoids duplicating player.js's fetch/parse logic.
  // Unchanged — this was never a Firestore/RTDB read to begin with.
  let latestEpisode = null;
  const firstCard = document.querySelector('#listenEpisodeList .episode-card');
  if (firstCard) {
    latestEpisode = {
      title: firstCard.querySelector('.episode-card-title')?.textContent?.trim() || null,
    };
  } else {
    watchForLatestEpisode();
  }

  return {
    learnProfile, effectiveStreak, missedDaysInRow, doneToday, activeCourse, wrongCount, xp,
    friendReqCount: finalFriendReqCount, friendsCount: finalFriendsCount, sharedCourseCount: finalSharedCourseCount,
    latestEpisode,
  };
}

// Fallback path used only when the RTDB mirror hasn't populated activeCourse yet.
async function loadActiveCourseFromFirestore(uidStr) {
  const coursesSnap = await getDocs(query(collection(db, 'users', uidStr, 'learnCourses'), orderBy('lastOpenedAt', 'desc'), limit(1))).catch(() => null);
  return coursesSnap && !coursesSnap.empty ? { id: coursesSnap.docs[0].id, ...coursesSnap.docs[0].data() } : null;
}

// Fallback path used only when the RTDB mirror hasn't populated wrongCount yet.
async function loadWrongCountFromFirestore(uidStr, courseId) {
  const wrongSnap = await getDocs(collection(db, 'users', uidStr, 'learnCourses', courseId, 'wrongAnswers')).catch(() => null);
  return wrongSnap ? wrongSnap.size : 0;
}

let episodeWatcherStarted = false;
function watchForLatestEpisode() {
  const list = document.getElementById('listenEpisodeList');
  if (!list || episodeWatcherStarted) return;
  episodeWatcherStarted = true;
  const observer = new MutationObserver(() => {
    if (list.querySelector('.episode-card')) {
      observer.disconnect();
      const homePage = document.getElementById('page-home');
      if (homePage && homePage.style.display !== 'none') refreshHome();
    }
  });
  observer.observe(list, { childList: true });
}

// Renders every section of Home from the fetched data. Each section decides
// for itself how to handle "however many items exist today" — the hero is
// always exactly one, stats is always a fixed 1-or-2-cell row, notifications
// is a plain list that grows/shrinks with zero layout math, and games is a
// horizontally-scrolling row. Nothing here has to be pre-sized against
// whatever else happens to be on the page, so there's no combination of
// present/absent optional items that leaves an empty gap.
function renderHome(data) {
  if (homeLoadingEl) homeLoadingEl.style.display = 'none';
  if (homeSkeletonEl) homeSkeletonEl.style.display = 'none';
  renderHomeHero(data);
  renderHomeStats(data);
  renderHomeUsageBanner();
  renderHomeNotifications(data);
  renderHomeGames();
}

// ---- Usage banner: free-tier "Games left / Lessons left" (from today's
// local usage counts), or a "Unlimited unlocked" callout for Premium.
// Re-run on every premium-status change too, not just on home renders,
// since purchasing/restoring can flip isPremium() without necessarily
// re-running renderHome() right away. ----
function renderHomeUsageBanner() {
  if (!homeUsageBannerEl) return;

  // Lesson/game gating no longer exists — everyone gets unlimited Games &
  // Lessons, free or Premium, so there's nothing left to count down. Keep
  // a lightweight banner rather than removing it outright since it's a
  // nice, cheap "you have everything" reassurance on Home.
  homeUsageBannerEl.style.display = 'flex';
  homeUsageBannerEl.className = isPremium() ? 'home-usage-banner c-premium' : 'home-usage-banner c-free';
  homeUsageBannerEl.innerHTML = `
    <span class="material-symbols-outlined">bolt</span>
    <span>Unlimited Games &amp; Lessons!</span>
  `;

  document.getElementById('homeGoPremiumBtn')?.classList.toggle('is-premium-member', isPremium());
  homeReviewPageBtn?.classList.toggle('is-premium-member', isPremium());
}

onPremiumChange(() => renderHomeUsageBanner());

// ---- Hero: continue course / start one / review / complete — the single
// dominant card, always exactly one, so it never competes for prominence
// with anything else on the page. ----
// Lightens (positive percent) or darkens (negative percent) a hex color by
// mixing it toward white/black. Used to turn a course's single accent color
// into the two-tone gradient the hero card is styled with.
function shadeHexColor(hex, percent) {
  const num = parseInt(hex.slice(1), 16);
  const amt = Math.round(2.55 * percent);
  const r = Math.max(0, Math.min(255, ((num >> 16) & 0xff) + amt));
  const g = Math.max(0, Math.min(255, ((num >> 8) & 0xff) + amt));
  const b = Math.max(0, Math.min(255, (num & 0xff) + amt));
  return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, '0')}`;
}

function renderHomeHero(data) {
  const { activeCourse } = data;
  if (!homeHeroEl) return;
  homeHeroEl.style.display = '';
  homeHeroEl.onclick = goToLearnTab;
  homeHeroProgressTrackEl.style.display = 'none';

  // Theme the card to the active course's color (falls back to the default
  // blue gradient via CSS when there's no course, or no color on it).
  homeHeroEl.style.background = activeCourse?.color
    ? `linear-gradient(135deg, ${shadeHexColor(activeCourse.color, 12)}, ${shadeHexColor(activeCourse.color, -12)})`
    : '';

  if (!activeCourse) {
    homeHeroLabelEl.textContent = 'Start learning';
    homeHeroTitleEl.textContent = 'New course';
    homeHeroSubEl.textContent = 'Type anything you want to learn';
    return;
  }

  const unitIndex = activeCourse.currentUnitIndex || 0;
  const courseDone = unitIndex >= 10;
  if (activeCourse.status === 'generating' && !courseDone) {
    homeHeroLabelEl.textContent = 'Preparing your lesson…';
    homeHeroTitleEl.textContent = activeCourse.title;
    homeHeroSubEl.textContent = 'Check back in a moment';
  } else if (courseDone && activeCourse.courseReviewCompleted) {
    homeHeroLabelEl.textContent = 'Course complete!';
    homeHeroTitleEl.textContent = activeCourse.title;
    homeHeroSubEl.textContent = 'Ready for something new?';
  } else if (courseDone) {
    homeHeroLabelEl.textContent = 'Final review ready';
    homeHeroTitleEl.textContent = activeCourse.title;
    homeHeroSubEl.textContent = '15 questions on the whole course';
  } else {
    homeHeroLabelEl.textContent = 'Continue where you left off';
    homeHeroTitleEl.textContent = activeCourse.title;
    homeHeroSubEl.textContent = `Unit ${unitIndex + 1}, lesson ${(activeCourse.currentLessonIndex || 0) + 1}`;
    homeHeroProgressTrackEl.style.display = '';
    homeHeroProgressFillEl.style.width = `${Math.min(100, Math.round((unitIndex / 10) * 100))}%`;
  }
}

// ---- Stats: a fixed 2x2 grid — streak, XP, friends, review — always the
// same four cells so there's no variable count to pack. Review swaps to a
// positive green "all caught up" state at 0 instead of hiding, so the grid
// shape never changes. ----
function renderHomeStats(data) {
  const { effectiveStreak, missedDaysInRow, doneToday, xp, friendsCount } = data;
  if (!homeStatsRowEl) return;
  homeStatsRowEl.style.display = '';
  if (homeXpShopBannerEl) homeXpShopBannerEl.style.display = '';

  homeStatStreakNumEl.textContent = String(effectiveStreak);
  homeStatStreakLabelEl.textContent = effectiveStreak === 0
    ? 'Start today'
    : (doneToday ? 'Day streak, done!' : (missedDaysInRow >= 1 ? "Don't lose it!" : 'Day streak'));
  homeStatStreakEl.onclick = (effectiveStreak > 0 && doneToday)
    ? async () => { await ensureLearnInitialized(); goToLearnTab(); document.getElementById('learnStreakBtn')?.click(); }
    : goToLearnTab;

  homeStatXpNumEl.textContent = String(xp);

  homeStatFriendsNumEl.textContent = String(friendsCount);
  homeStatFriendsEl.onclick = () => profileFriendsBtn?.click();

  // Weak-spot count read straight from learn.js's own in-memory state
  // (kept live by maybeGenerateReviewSpots()) rather than the RTDB
  // mirror's wrongCount, which now only feeds Daily Lesson's mistake-mix,
  // not this stat card.
  const weakSpotCount = getWeakSpotCount();
  homeStatReviewEl.classList.remove('c-error', 'c-green');
  if (weakSpotCount > 0) {
    homeStatReviewEl.classList.add('c-error');
    homeStatReviewNumEl.textContent = String(weakSpotCount);
    homeStatReviewLabelEl.textContent = 'Weak Spots to Review';
    homeStatReviewEl.onclick = async () => { await ensureLearnInitialized(); goToLearnTab(); openReviewPage(); };
  } else {
    homeStatReviewEl.classList.add('c-green');
    homeStatReviewNumEl.textContent = '0';
    homeStatReviewLabelEl.textContent = 'All caught up!';
    homeStatReviewEl.onclick = goToLearnTab;
  }
}

// ---- Notifications: a plain vertical list — friend requests, shared
// courses, latest episode. 0 items hides the section, N items is just N
// rows; the list absorbs the count instead of a grid trying to. ----
function renderHomeNotifications(data) {
  const { friendReqCount, sharedCourseCount, latestEpisode } = data;
  if (!homeNotifListEl) return;
  homeNotifListEl.innerHTML = '';

  const items = [];
  if (friendReqCount > 0) {
    items.push({
      icon: 'group_add', color: 'purple',
      text: `${friendReqCount} friend request${friendReqCount === 1 ? '' : 's'}`,
      onClick: () => profileFriendsBtn?.click(),
    });
  }
  if (sharedCourseCount > 0) {
    items.push({
      icon: 'card_giftcard', color: 'pink',
      text: `${sharedCourseCount} course${sharedCourseCount === 1 ? '' : 's'} shared with you`,
      onClick: async () => { await ensureLearnInitialized(); goToLearnTab(); },
    });
  }
  items.push({
    icon: 'podcasts', color: 'teal',
    text: latestEpisode?.title ? latestEpisode.title : 'Catch up on the podcast',
    onClick: () => document.querySelector('.nav-btn[data-page="listen"]')?.click(),
  });

  homeNotifSectionEl.style.display = items.length ? '' : 'none';
  items.forEach((item) => {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = `home-notif-row c-${item.color}`;
    row.innerHTML = `
      <span class="material-symbols-outlined home-notif-icon">${item.icon}</span>
      <div class="home-notif-text">${escapeHtmlMain(item.text)}</div>
      <span class="material-symbols-outlined home-notif-chevron">chevron_right</span>
    `;
    row.addEventListener('click', item.onClick);
    homeNotifListEl.appendChild(row);
  });
}

// ---- Games: a horizontally-scrolling row showing every Learning Games
// entry (Join Game plus all 8 games), so nothing is left undiscoverable
// behind the Games page. Each card just re-triggers the same button the
// Learning Games page itself uses, so behavior never drifts out of sync.
function renderHomeGames() {
  if (!homeGamesRowEl) return;
  homeGamesRowEl.innerHTML = '';

  homeGamesSectionEl.style.display = HOME_GAME_LIST.length ? '' : 'none';
  HOME_GAME_LIST.forEach((game) => {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = `home-game-card c-${game.color}`;
    card.innerHTML = `
      <div class="home-game-title">${escapeHtmlMain(game.name)}</div>
      <div class="home-game-sub">${escapeHtmlMain(game.desc)}</div>
    `;
    card.addEventListener('click', async () => { await ensureLearnInitialized(); document.getElementById(game.cardId)?.click(); });
    homeGamesRowEl.appendChild(card);
  });
}

// ============================================================
// ============================================================
// AVATAR (1–2 emoji on a chosen background color)
// ============================================================
let currentAvatar = null; // { emoji, color } or null

// Renders an avatar into any of the avatar-display elements (settings,
// friends list rows, friend profile, etc). `avatar` is { emoji, color } or
// null/undefined, in which case a generic person icon is shown.
function renderAvatarInto(el, avatar) {
  if (avatar && avatar.emoji) {
    el.style.background = avatar.color || '';
    const chars = Array.from(avatar.emoji);
    el.innerHTML = `<span class="avatar-emoji" style="font-size:${chars.length >= 2 ? '0.62em' : '1em'};">${avatar.emoji}</span>`;
  } else {
    el.style.background = '';
    el.innerHTML = `<span class="material-symbols-outlined">person</span>`;
  }
}

// Curated, kid-friendly emoji choices for the avatar picker.
const AVATAR_EMOJI_CHOICES = [
  '🦁', '🐶', '🐱', '🐸', '🦊', '🐼', '🐵', '🦄', '🐬', '🦋', '🐝', '🐢', '🦖', '🐙', '🐰', '🦉',
  '🚀', '⭐', '🌈', '🔥', '⚡', '🎨', '🎮', '📚', '🎵', '⚽', '🏀', '🎸', '🍕', '🍦',
];

// 15 preset background colors for the avatar picker.
const AVATAR_COLOR_CHOICES = [
  '#FF6B6B', '#FF8A2B', '#FFC93C', '#6BCB77', '#34D8A6',
  '#2BC5C5', '#4FA6FF', '#1E6FE0', '#123B7A', '#7C6BFF',
  '#B36BFF', '#FF6BCB', '#FF3B6E', '#C77D3B', '#8D99AE',
];

const avatarModalOverlay = document.getElementById('avatarModalOverlay');
const avatarPreview = document.getElementById('avatarPreview');
const avatarEmojiGrid = document.getElementById('avatarEmojiGrid');
const avatarColorGrid = document.getElementById('avatarColorGrid');
const avatarError = document.getElementById('avatarError');
const avatarSaveBtn = document.getElementById('avatarSaveBtn');
const avatarCancelBtn = document.getElementById('avatarCancelBtn');

let avatarSelectedEmoji = []; // up to 2 emoji strings, in pick order
let avatarSelectedColor = AVATAR_COLOR_CHOICES[0];

// Rebuilds the emoji/color grids from the 30/15 defaults PLUS whatever
// this account has unlocked from the XP Shop — called once at startup and
// again any time onShopStateChange fires with a different unlocked list,
// so a purchase made from the shop shows up here without needing to
// reopen the picker.
let _lastRenderedUnlockedEmoji = [];
let _lastRenderedUnlockedColors = [];
function rebuildAvatarPickerGrids(state) {
  const emojiChoices = [...AVATAR_EMOJI_CHOICES, ...(state.unlockedEmoji || [])];
  const colorChoices = [...AVATAR_COLOR_CHOICES, ...(state.unlockedColors || [])];
  if (
    emojiChoices.length === avatarEmojiGrid.childElementCount
    && colorChoices.length === avatarColorGrid.childElementCount
    && (state.unlockedEmoji || []).join('') === _lastRenderedUnlockedEmoji.join('')
    && (state.unlockedColors || []).join('') === _lastRenderedUnlockedColors.join('')
  ) return; // nothing new to add — skip the rebuild

  _lastRenderedUnlockedEmoji = state.unlockedEmoji || [];
  _lastRenderedUnlockedColors = state.unlockedColors || [];

  avatarEmojiGrid.innerHTML = '';
  emojiChoices.forEach((emoji) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'avatar-emoji-btn';
    btn.textContent = emoji;
    btn.dataset.emoji = emoji;
    btn.addEventListener('click', () => {
      const idx = avatarSelectedEmoji.indexOf(emoji);
      if (idx !== -1) {
        avatarSelectedEmoji.splice(idx, 1);
      } else {
        if (avatarSelectedEmoji.length >= 2) avatarSelectedEmoji.shift(); // drop oldest pick
        avatarSelectedEmoji.push(emoji);
      }
      avatarError.textContent = '';
      refreshAvatarModalUI();
    });
    avatarEmojiGrid.appendChild(btn);
  });

  avatarColorGrid.innerHTML = '';
  colorChoices.forEach((color) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'avatar-color-swatch';
    btn.style.background = color;
    btn.dataset.color = color;
    btn.addEventListener('click', () => {
      avatarSelectedColor = color;
      refreshAvatarModalUI();
    });
    avatarColorGrid.appendChild(btn);
  });

  refreshAvatarModalUI();
}
onShopStateChange(rebuildAvatarPickerGrids);

function refreshAvatarModalUI() {
  avatarEmojiGrid.querySelectorAll('.avatar-emoji-btn').forEach((btn) => {
    btn.classList.toggle('selected', avatarSelectedEmoji.includes(btn.dataset.emoji));
  });
  avatarColorGrid.querySelectorAll('.avatar-color-swatch').forEach((btn) => {
    btn.classList.toggle('selected', btn.dataset.color === avatarSelectedColor);
  });
  renderAvatarInto(avatarPreview, {
    emoji: avatarSelectedEmoji.join(''),
    color: avatarSelectedColor,
  });
}

profileAvatarBtn.addEventListener('click', () => {
  avatarError.textContent = '';
  avatarSelectedEmoji = currentAvatar?.emoji ? Array.from(currentAvatar.emoji).slice(0, 2) : [];
  avatarSelectedColor = currentAvatar?.color || AVATAR_COLOR_CHOICES[0];
  refreshAvatarModalUI();
  avatarModalOverlay.classList.add('show');
  maybeShowOverlay('avatarCustomizeWelcome');
});
avatarCancelBtn.addEventListener('click', () => avatarModalOverlay.classList.remove('show'));

avatarSaveBtn.addEventListener('click', async () => {
  if (!avatarSelectedEmoji.length) {
    avatarError.textContent = 'Pick at least 1 emoji.';
    return;
  }
  avatarSaveBtn.disabled = true;
  avatarSaveBtn.textContent = 'Saving…';
  avatarError.textContent = '';
  try {
    const u = auth.currentUser;
    if (!u) throw new Error('Not signed in');
    const avatar = { emoji: avatarSelectedEmoji.join(''), color: avatarSelectedColor };
    await rtdbSet(rtdbRef(rtdb, `avatars/${u.uid}`), avatar);
    currentAvatar = avatar;
    renderAvatarInto(settingsAvatar, currentAvatar);
    avatarModalOverlay.classList.remove('show');
  } catch (err) {
    avatarError.textContent = err.message || 'Could not save avatar.';
  } finally {
    avatarSaveBtn.disabled = false;
    avatarSaveBtn.textContent = 'Save Avatar';
  }
});

// ============================================================
// XP + PUBLIC PROFILE SYNC
// ============================================================

// Upserts the minimal email-lookup doc and the public-facing profile
// fields. Safe to call every time the Profile page loads — merge:true means
// it never clobbers xp/streak/photo already written elsewhere.
async function syncPublicProfileDocs(user) {
  if (!user) return;
  try {
    await setDoc(doc(db, 'userDirectory', user.uid), { email: (user.email || '').toLowerCase() }, { merge: true });
    await setDoc(doc(db, 'userProfiles', user.uid), {
      displayName: user.displayName || '',
      email: user.email || '',
      premium: isPremium(),
    }, { merge: true });
  } catch (err) {
    console.error('Failed to sync public profile docs:', err);
  }
}
// Keep it in sync going forward too, not just at the moments
// syncPublicProfileDocs() happens to get called from.
onPremiumChange((premium) => {
  const u = auth.currentUser;
  if (!u) return;
  setDoc(doc(db, 'userProfiles', u.uid), { premium }, { merge: true }).catch((err) => {
    console.error('Failed to sync premium flag to public profile:', err);
  });
});

async function loadXpTotal() {
  const u = auth.currentUser;
  if (!u) return;
  try {
    const snap = await getDoc(doc(db, 'users', u.uid, 'learnProfile', 'main'));
    const xp = snap.exists() ? (snap.data().xp || 0) : 0;
    profileXpCount.textContent = xp;
  } catch (err) {
    console.error('Failed to load XP total:', err);
  }
}

// A monotonically increasing token so that if this gets called again (e.g.
// the learner mashes the Settings button, or switches accounts) before an
// earlier call's Firestore read has resolved, the earlier call's result —
// which could be stale or for the wrong account — can recognize it's no
// longer the latest request and skip updating the DOM.
let avatarLoadToken = 0;

// Always re-fetches the avatar from RTDB rather than trusting the
// in-memory `currentAvatar` cache, so every time the Settings/Profile page
// is opened it reflects what's actually saved — previously this was only
// ever refreshed as a side effect of loadXpTotal(), so a slow or failed
// read (or an out-of-order response from an earlier call) could leave the
// picture blank or stale.
async function loadAvatarForCurrentUser() {
  const u = auth.currentUser;
  if (!u) return;
  const myToken = ++avatarLoadToken;
  try {
    const avatarSnap = await rtdbGet(rtdbRef(rtdb, `avatars/${u.uid}`));
    if (myToken !== avatarLoadToken) return; // a newer call has since started — don't clobber its result
    const avatarData = avatarSnap.exists() ? avatarSnap.val() : null;
    currentAvatar = avatarData && avatarData.emoji ? { emoji: avatarData.emoji, color: avatarData.color } : null;
    renderAvatarInto(settingsAvatar, currentAvatar);
  } catch (err) {
    console.error('Failed to load avatar:', err);
  }
}

// Live-updates the Profile page's XP count the instant XP is earned
// elsewhere in the app (e.g. finishing a lesson in learn.js), instead of
// only reflecting it the next time the Profile page happens to reload.
let unsubscribeXp = null;
function startXpListener(user) {
  stopXpListener();
  if (!user) return;
  unsubscribeXp = onSnapshot(doc(db, 'users', user.uid, 'learnProfile', 'main'), (snap) => {
    const xp = snap.exists() ? (snap.data().xp || 0) : 0;
    profileXpCount.textContent = xp;
    if (friendsPageOverlay?.classList.contains('show')) {
      friendsMyXp.textContent = xp;
    }
  }, (err) => {
    console.error('XP listener failed:', err);
  });
}
function stopXpListener() {
  if (unsubscribeXp) {
    unsubscribeXp();
    unsubscribeXp = null;
  }
}

// ============================================================
// NOTIFICATIONS (bell icon, in-app panel, push registration)
// ============================================================
// This app already talks to Cloudflare Workers for email — the push side
// of this reuses that same pattern instead of Firebase Cloud Functions, so
// no Blaze/billing plan is needed. See notifications-worker/README.md.

// Paste your Web Push certificate ("VAPID key") from
// Firebase Console → Project Settings → Cloud Messaging → Web configuration.
const FCM_VAPID_KEY = 'BJs13nmUj_6h8LK0W8z3mj1VXGRqbw_38lTsu40KHDgQLAPz5U1oOO0U3p1TIuOTgOl-PFQcppTefgCkuZBSr9I';

const notifBellBtn = document.getElementById('notifBellBtn');
const notifBellBadge = document.getElementById('notifBellBadge');
const notifPanelOverlay = document.getElementById('notifPanelOverlay');
const notifPanelExitBtn = document.getElementById('notifPanelExitBtn');
const notifPanelList = document.getElementById('notifPanelList');
const notifPanelEmpty = document.getElementById('notifPanelEmpty');

const NOTIF_ICONS = {
  friend_request: 'person_add',
  friend_accepted: 'how_to_reg',
  course_shared: 'card_giftcard',
  streak_reminder: 'local_fire_department',
  new_episode: 'podcasts',
};

let unsubscribeNotifs = null;
let pushInitDone = false;
let latestNotifs = []; // mirrors the last onSnapshot payload, so opening the panel knows what to mark read

function startNotifBell(user) {
  stopNotifBell();
  const q = query(collection(db, 'users', user.uid, 'notifications'), orderBy('createdAt', 'desc'), limit(30));
  unsubscribeNotifs = onSnapshot(q, (snap) => {
    const notifs = snap.docs.map((d) => ({ id: d.id, ...d.data() }));
    latestNotifs = notifs;
    renderNotifBadge(notifs);
    renderNotifPanel(notifs);
  }, (err) => console.error('Notifications listener failed:', err));
}
function stopNotifBell() {
  if (unsubscribeNotifs) { unsubscribeNotifs(); unsubscribeNotifs = null; }
  notifBellBadge.style.display = 'none';
  notifPanelList.innerHTML = '';
  latestNotifs = [];
  pushInitDone = false;
}

function renderNotifBadge(notifs) {
  const unread = notifs.filter((n) => !n.read).length;
  if (unread > 0) {
    notifBellBadge.textContent = unread > 9 ? '9+' : String(unread);
    notifBellBadge.style.display = 'flex';
  } else {
    notifBellBadge.style.display = 'none';
  }
}

function timeAgoNotif(ms) {
  const diff = Date.now() - ms;
  const min = Math.floor(diff / 60000);
  if (min < 1) return 'just now';
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  return `${Math.floor(hr / 24)}d ago`;
}

function renderNotifPanel(notifs) {
  notifPanelList.innerHTML = '';
  notifPanelEmpty.style.display = notifs.length ? 'none' : '';
  notifs.forEach((n) => {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = `notif-row${n.read ? '' : ' unread'}`;
    row.innerHTML = `
      <div class="notif-row-icon"><span class="material-symbols-outlined">${NOTIF_ICONS[n.type] || 'notifications'}</span></div>
      <div>
        <div class="notif-row-title">${escapeHtmlMain(n.title || '')}</div>
        ${n.body ? `<div class="notif-row-body">${escapeHtmlMain(n.body)}</div>` : ''}
        <div class="notif-row-time">${timeAgoNotif(n.createdAt || Date.now())}</div>
      </div>
    `;
    row.addEventListener('click', () => onNotifRowTap(n));
    notifPanelList.appendChild(row);
  });
}

async function onNotifRowTap(n) {
  const u = auth.currentUser;
  if (u && !n.read) {
    updateDoc(doc(db, 'users', u.uid, 'notifications', n.id), { read: true }).catch(() => {});
  }
  await routeForNotifType(n.type);
}

// Marks every currently-unread notification as read in one batched write —
// called whenever the panel is opened, so the bell badge count actually
// clears instead of staying stuck at whatever it first showed (it
// previously only ever cleared per-notification, via onNotifRowTap above,
// which meant notifications the person never individually tapped kept
// counting forever).
async function markAllNotifsRead() {
  const u = auth.currentUser;
  if (!u) return;
  const unread = latestNotifs.filter((n) => !n.read);
  if (!unread.length) return;
  try {
    const batch = writeBatch(db);
    unread.forEach((n) => {
      batch.update(doc(db, 'users', u.uid, 'notifications', n.id), { read: true });
    });
    await batch.commit();
  } catch (err) {
    console.error('Failed to mark notifications read:', err);
  }
}

// Shared "go to the right screen" logic for a notification type. Used both
// when tapping an in-app bell row (onNotifRowTap above) and when tapping a
// native OS push notification on Capacitor (see notificationActionPerformed
// listener in initNativePush below), since the latter only has the
// data.type payload to go on, not a full Firestore notification doc.
async function routeForNotifType(type) {
  if (type === 'friend_request' || type === 'friend_accepted') {
    notifPanelOverlay.classList.remove('show');
    profileFriendsBtn?.click();
  } else if (type === 'course_shared') {
    notifPanelOverlay.classList.remove('show');
    await ensureLearnInitialized();
    goToLearnTab();
  } else if (type === 'streak_reminder') {
    notifPanelOverlay.classList.remove('show');
    goToLearnTab();
  } else if (type === 'new_episode') {
    notifPanelOverlay.classList.remove('show');
    document.querySelector('.nav-btn[data-page="listen"]')?.click();
  }
}

notifBellBtn.addEventListener('click', () => {
  notifPanelOverlay.classList.add('show');
  initPushForCurrentUser(); // lazy: only asks for permission once the person shows interest
  markAllNotifsRead();
});
notifPanelExitBtn.addEventListener('click', () => notifPanelOverlay.classList.remove('show'));

// Registers this device for push, saved under the user's own pushTokens
// subcollection. Safe to call repeatedly — it no-ops once done, and does
// nothing if the person denies/ignores the permission prompt (they still
// see in-app notifications either way).
//
// Two very different code paths share this entry point:
//   - Native (Capacitor iOS/Android): uses the @capacitor-firebase/messaging
//     plugin, which talks to the native Firebase SDK and hands back a real
//     FCM token directly — no service worker involved (WKWebView doesn't
//     run firebase-messaging-sw.js the way a real browser tab does).
//   - Web (desktop/mobile browser, incl. "Add to Home Screen" on iOS 16.4+):
//     the original firebase/messaging web-SDK + service-worker flow.
// Both paths write to the same `users/{uid}/pushTokens` collection, so the
// Cloudflare Worker's sendFcm() doesn't need to know or care which one a
// given token came from.
async function initPushForCurrentUser() {
  if (pushInitDone) return;
  pushInitDone = true;
  const u = auth.currentUser;
  if (!u) return;
  try {
    if (window.Capacitor?.isNativePlatform?.()) {
      await initNativePush(u);
    } else {
      await initWebPush(u);
    }
  } catch (err) {
    console.warn('Push registration skipped:', err);
  }
}

async function initNativePush(u) {
  const FirebaseMessaging = window.Capacitor?.Plugins?.FirebaseMessaging;
  if (!FirebaseMessaging) {
    console.warn('FirebaseMessaging native plugin not found — did you run `npm install @capacitor-firebase/messaging && npx cap sync`?');
    return;
  }

  const current = await FirebaseMessaging.checkPermissions();
  let granted = current.receive === 'granted';
  if (!granted) {
    const requested = await FirebaseMessaging.requestPermissions();
    granted = requested.receive === 'granted';
  }
  if (!granted) return;

  const { token } = await FirebaseMessaging.getToken();
  if (!token) return;

  await setDoc(doc(db, 'users', u.uid, 'pushTokens', token), {
    token, createdAt: Date.now(), ua: 'capacitor-ios',
  });

  // If the OS later rotates the token, keep Firestore in sync.
  FirebaseMessaging.addListener('tokenReceived', (event) => {
    const newToken = event?.token;
    if (!newToken) return;
    setDoc(doc(db, 'users', u.uid, 'pushTokens', newToken), {
      token: newToken, createdAt: Date.now(), ua: 'capacitor-ios',
    }).catch(() => {});
  });

  // Foreground push (app open) — show the same in-app toast the web path
  // uses, since a foreground push doesn't surface as an OS banner.
  FirebaseMessaging.addListener('notificationReceived', (event) => {
    const n = event?.notification;
    showNotifToast(n?.title, n?.body, n?.data?.type);
  });

  // Tapping the OS notification banner (app backgrounded/closed) — route to
  // the relevant screen, same as tapping a row in the in-app bell panel.
  FirebaseMessaging.addListener('notificationActionPerformed', (event) => {
    const type = event?.notification?.data?.type;
    if (type) routeForNotifType(type);
  });
}

async function initWebPush(u) {
  if (!('Notification' in window) || !('serviceWorker' in navigator)) return;
  if (!(await isMessagingSupported())) return;
  const permission = await Notification.requestPermission();
  if (permission !== 'granted') return;

  const registration = await navigator.serviceWorker.register('./firebase-messaging-sw.js');
  const messaging = getMessaging();
  const token = await getToken(messaging, { vapidKey: FCM_VAPID_KEY, serviceWorkerRegistration: registration });
  if (!token) return;

  await setDoc(doc(db, 'users', u.uid, 'pushTokens', token), {
    token, createdAt: Date.now(), ua: navigator.userAgent,
  });

  // Foreground messages don't trigger the OS notification UI on their own —
  // show a small in-app toast instead so pushes are visible either way.
  onMessage(messaging, (payload) => {
    showNotifToast(payload.notification?.title, payload.notification?.body, payload.data?.type);
  });
}

let notifToastTimer = null;
function showNotifToast(title, body, type) {
  let toast = document.getElementById('notifToastEl');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'notifToastEl';
    toast.className = 'notif-toast';
    document.body.appendChild(toast);
  }
  toast.innerHTML = `
    <div class="notif-toast-icon"><span class="material-symbols-outlined">${NOTIF_ICONS[type] || 'notifications'}</span></div>
    <div>
      <div class="notif-toast-title">${escapeHtmlMain(title || 'Notification')}</div>
      ${body ? `<div class="notif-toast-body">${escapeHtmlMain(body)}</div>` : ''}
    </div>
  `;
  requestAnimationFrame(() => toast.classList.add('show'));
  clearTimeout(notifToastTimer);
  notifToastTimer = setTimeout(() => toast.classList.remove('show'), 4500);
}

// ============================================================
// FRIENDS
// ============================================================
const friendsPageOverlay = document.getElementById('friendsPageOverlay');
const friendsExitBtn = document.getElementById('friendsExitBtn');
const friendsMyAvatar = document.getElementById('friendsMyAvatar');
const friendsMyName = document.getElementById('friendsMyName');
const friendsMyEmail = document.getElementById('friendsMyEmail');
const friendsMyXp = document.getElementById('friendsMyXp');
const friendsMyStreak = document.getElementById('friendsMyStreak');
const addFriendOpenBtn = document.getElementById('addFriendOpenBtn');
const myQrOpenBtn = document.getElementById('myQrOpenBtn');
const scanAddFriendBtn = document.getElementById('scanAddFriendBtn');
const myQrModalOverlay = document.getElementById('myQrModalOverlay');
const myQrCanvas = document.getElementById('myQrCanvas');
const myQrCloseBtn = document.getElementById('myQrCloseBtn');
const myQrCourseDetectedEl = document.getElementById('myQrCourseDetected');
const blockOpenBtn = document.getElementById('blockOpenBtn');
const friendRequestsSection = document.getElementById('friendRequestsSection');
const friendRequestsList = document.getElementById('friendRequestsList');
const sentRequestsSection = document.getElementById('sentRequestsSection');
const sentRequestsList = document.getElementById('sentRequestsList');
const friendsList = document.getElementById('friendsList');
const friendsListEmpty = document.getElementById('friendsListEmpty');

const addFriendModalOverlay = document.getElementById('addFriendModalOverlay');
const addFriendEmailInput = document.getElementById('addFriendEmailInput');
const addFriendError = document.getElementById('addFriendError');
const addFriendSendBtn = document.getElementById('addFriendSendBtn');
const addFriendCancelBtn = document.getElementById('addFriendCancelBtn');

const blockModalOverlay = document.getElementById('blockModalOverlay');
const blockEmailInput = document.getElementById('blockEmailInput');
const blockError = document.getElementById('blockError');
const blockConfirmBtn = document.getElementById('blockConfirmBtn');
const blockCancelBtn = document.getElementById('blockCancelBtn');

const friendProfilePageOverlay = document.getElementById('friendProfilePageOverlay');
const friendProfileExitBtn = document.getElementById('friendProfileExitBtn');
const friendProfileTitle = document.getElementById('friendProfileTitle');
const friendProfileAvatar = document.getElementById('friendProfileAvatar');
const friendProfileName = document.getElementById('friendProfileName');
const friendProfileEmail = document.getElementById('friendProfileEmail');
const friendProfileXp = document.getElementById('friendProfileXp');
const friendProfileStreak = document.getElementById('friendProfileStreak');
const friendProfileRemoveBtn = document.getElementById('friendProfileRemoveBtn');
const friendProfileBadges = document.getElementById('friendProfileBadges');
const friendProfileFriendsEmpty = document.getElementById('friendProfileFriendsEmpty');
const friendProfileFriendsList = document.getElementById('friendProfileFriendsList');

let openFriendProfileUid = null; // whoever's profile is currently open in the friend-profile fullpager

profileFriendsBtn.addEventListener('click', () => {
  friendsPageOverlay.classList.add('show');
  loadFriendsPage();
  maybeShowOverlay('addFriendsWelcome');
});
friendsExitBtn.addEventListener('click', () => friendsPageOverlay.classList.remove('show'));

async function loadFriendsPage() {
  const u = auth.currentUser;
  if (!u) return;

  // ---- My own summary header ----
  renderAvatarInto(friendsMyAvatar, currentAvatar);
  friendsMyName.textContent = formatDisplayName(u.displayName || (u.email ? u.email.split('@')[0] : 'Learner'), isPremium());
  friendsMyEmail.textContent = u.email || '';

  // Kicked off in parallel, not awaited before it starts — this is what
  // gets the cached/skeleton friend rows painting instantly instead of
  // sitting behind this header's own Firestore round-trip first.
  const friendsPromise = loadFriendsAndRequests();

  try {
    const learnSnap = await getDoc(doc(db, 'users', u.uid, 'learnProfile', 'main'));
    const data = learnSnap.exists() ? learnSnap.data() : {};
    friendsMyXp.textContent = data.xp || 0;
    friendsMyStreak.textContent = data.streak || 0;
  } catch { /* leave as-is on failure */ }

  await friendsPromise;
}

// ---- Friends list cache (instant render while the real Firestore read
// happens in the background — see loadFriendsAndRequests below) ----
// Keyed per-uid so switching accounts on the same device can't bleed one
// person's cached friends into another's list, even for a flash of a frame.
const FRIENDS_CACHE_PREFIX = 'kll_friends_cache_';
function friendsCacheKey(uid) { return `${FRIENDS_CACHE_PREFIX}${uid}`; }
function readFriendsCache(uid) {
  try {
    const raw = localStorage.getItem(friendsCacheKey(uid));
    const parsed = raw ? JSON.parse(raw) : null;
    return Array.isArray(parsed) ? parsed : null;
  } catch { return null; }
}
function writeFriendsCache(uid, entries) {
  try { localStorage.setItem(friendsCacheKey(uid), JSON.stringify(entries)); } catch { /* best-effort only */ }
}

// Builds one accepted-friend row. Pulled out of loadFriendsAndRequests so
// the exact same markup/handlers back both the instant cached render and
// the real render once the background fetch resolves.
function buildFriendRow(uid, info) {
  const row = document.createElement('div');
  row.className = 'friend-row';
  row.innerHTML = `
    ${miniAvatarHtml(info)}
    <button type="button" class="friend-row-tap" data-uid="${uid}">
      <div class="friend-row-name">${escapeHtmlMain(formatDisplayName(info.displayName || info.email || 'Learner', !!info.premium))}</div>
      <div class="friend-row-sub">${info.xp || 0} XP · ${info.streak || 0} day streak</div>
    </button>
    <button type="button" class="friend-row-btn" data-uid="${uid}">Remove Friend</button>
  `;
  row.querySelector('.friend-row-tap').addEventListener('click', () => openFriendProfile(uid));
  row.querySelector('.friend-row-btn').addEventListener('click', () => removeFriend(uid, false));
  return row;
}

// Skeleton placeholder rows shown only when there's no cache yet to render
// instantly (e.g. this device's very first time opening Friends).
function renderFriendsSkeleton(count = 3) {
  friendsListEmpty.style.display = 'none';
  friendsList.innerHTML = Array.from({ length: count }).map(() => `
    <div class="friend-row friend-row-skeleton">
      <div class="skeleton-block skeleton-avatar"></div>
      <div class="friend-row-tap" style="cursor:default;">
        <div class="skeleton-block skeleton-line-name"></div>
        <div class="skeleton-block skeleton-line-sub"></div>
      </div>
    </div>
  `).join('');
}

async function loadFriendsAndRequests() {
  const u = auth.currentUser;
  if (!u) return;
  friendRequestsList.innerHTML = '';
  sentRequestsList.innerHTML = '';
  friendRequestsSection.style.display = 'none';
  sentRequestsSection.style.display = 'none';

  // Instant paint from cache (if we have any) so the friends list never
  // shows blank while the real Firestore round-trip below is in flight;
  // falls back to skeleton rows only when there's nothing cached yet.
  const cachedFriends = readFriendsCache(u.uid);
  if (cachedFriends && cachedFriends.length) {
    friendsListEmpty.style.display = 'none';
    friendsList.innerHTML = '';
    cachedFriends.forEach(({ uid, info }) => friendsList.appendChild(buildFriendRow(uid, info)));
  } else {
    renderFriendsSkeleton();
  }

  let snap;
  try {
    snap = await getDocs(collection(db, 'users', u.uid, 'friends'));
  } catch (err) {
    console.error('Failed to load friends:', err);
    return;
  }

  const accepted = [];
  const incoming = [];
  const outgoing = [];
  snap.forEach((d) => {
    const data = d.data();
    if (data.status === 'accepted') accepted.push({ uid: d.id, ...data });
    else if (data.status === 'pending' && data.direction === 'received') incoming.push({ uid: d.id, ...data });
    else if (data.status === 'pending' && data.direction === 'sent') outgoing.push({ uid: d.id, ...data });
  });

  if (incoming.length) {
    friendRequestsSection.style.display = '';
    for (const req of incoming) {
      const info = await fetchMiniProfile(req.uid);
      const row = document.createElement('div');
      row.className = 'friend-row';
      row.innerHTML = `
        ${miniAvatarHtml(info)}
        <div class="friend-row-tap" style="cursor:default;">
          <div class="friend-row-name">${escapeHtmlMain(formatDisplayName(info.displayName || info.email || 'Learner', !!info.premium))}</div>
          <div class="friend-row-sub">${escapeHtmlMain(info.email || '')}</div>
        </div>
        <div class="friend-row-btns">
          <button type="button" class="friend-row-btn accept" data-uid="${req.uid}">Accept</button>
          <button type="button" class="friend-row-btn" data-uid="${req.uid}">Decline</button>
        </div>
      `;
      const [acceptBtn, declineBtn] = row.querySelectorAll('.friend-row-btn');
      acceptBtn.addEventListener('click', () => respondToFriendRequest(req.uid, true));
      declineBtn.addEventListener('click', () => respondToFriendRequest(req.uid, false));
      friendRequestsList.appendChild(row);
    }
  }

  if (outgoing.length) {
    sentRequestsSection.style.display = '';
    for (const req of outgoing) {
      const info = await fetchMiniProfile(req.uid);
      const row = document.createElement('div');
      row.className = 'friend-row';
      row.innerHTML = `
        ${miniAvatarHtml(info)}
        <div class="friend-row-tap" style="cursor:default;">
          <div class="friend-row-name">${escapeHtmlMain(formatDisplayName(info.displayName || info.email || 'Learner', !!info.premium))}</div>
          <div class="friend-row-sub">${escapeHtmlMain(info.email || '')}</div>
        </div>
        <div class="friend-row-btns">
          <button type="button" class="friend-row-btn pending" disabled>Request Pending</button>
          <button type="button" class="friend-row-btn" data-uid="${req.uid}">Cancel</button>
        </div>
      `;
      row.querySelector('.friend-row-btn:not(.pending)').addEventListener('click', () => cancelFriendRequest(req.uid));
      sentRequestsList.appendChild(row);
    }
  }

  checkFriendBadges(accepted.length);

  if (!accepted.length) {
    friendsList.innerHTML = '';
    friendsListEmpty.style.display = '';
    writeFriendsCache(u.uid, []);
  } else {
    friendsList.innerHTML = ''; // replace cache/skeleton now that live data is ready
    const freshCacheEntries = [];
    for (const friend of accepted) {
      const info = await fetchMiniProfile(friend.uid);
      freshCacheEntries.push({ uid: friend.uid, info });
      friendsList.appendChild(buildFriendRow(friend.uid, info));
    }
    writeFriendsCache(u.uid, freshCacheEntries);
  }
}

// Reads a friend/candidate's public profile doc. Only succeeds under the
// Firestore rules if we're either the owner or an accepted friend — for a
// pending *incoming* request the sender's profile isn't readable yet, so we
// fall back to userDirectory (email only) in that case.
async function fetchMiniProfile(otherUid) {
  let info = null;
  try {
    const snap = await getDoc(doc(db, 'userProfiles', otherUid));
    if (snap.exists()) info = snap.data();
  } catch { /* likely not-yet-accepted — fall through */ }
  if (!info) {
    try {
      const dirSnap = await getDoc(doc(db, 'userDirectory', otherUid));
      if (dirSnap.exists()) info = { email: dirSnap.data().email };
    } catch { /* ignore */ }
  }
  if (!info) return {};
  try {
    const avatarSnap = await rtdbGet(rtdbRef(rtdb, `avatars/${otherUid}`));
    if (avatarSnap.exists()) {
      const avatarData = avatarSnap.val();
      info.avatarEmoji = avatarData.emoji;
      info.avatarColor = avatarData.color;
    }
  } catch { /* not readable / no avatar set — leave unset */ }
  return info;
}

function escapeHtmlMain(str) {
  return (str || '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// Builds the small avatar chip markup used in friend list rows, from a mini
// profile's avatarEmoji/avatarColor fields (falls back to a generic person
// icon when no avatar has been set).
function miniAvatarHtml(info) {
  if (info && info.avatarEmoji) {
    const chars = Array.from(info.avatarEmoji);
    const fontSize = chars.length >= 2 ? '0.62em' : '1em';
    return `<div class="settings-avatar friend-row-avatar" style="background:${info.avatarColor || ''};"><span class="avatar-emoji" style="font-size:${fontSize};">${escapeHtmlMain(info.avatarEmoji)}</span></div>`;
  }
  return `<div class="settings-avatar friend-row-avatar"><span class="material-symbols-outlined">person</span></div>`;
}

// ---- Add Friend ----
addFriendOpenBtn.addEventListener('click', () => {
  addFriendEmailInput.value = '';
  addFriendError.textContent = '';
  addFriendModalOverlay.classList.add('show');
});
addFriendCancelBtn.addEventListener('click', () => addFriendModalOverlay.classList.remove('show'));

addFriendSendBtn.addEventListener('click', async () => {
  const email = addFriendEmailInput.value.trim().toLowerCase();
  const u = auth.currentUser;
  if (!u) return;
  if (!email || !email.includes('@')) { addFriendError.textContent = 'Enter a valid email address.'; return; }
  if (email === (u.email || '').toLowerCase()) { addFriendError.textContent = "That's your own email!"; return; }

  addFriendSendBtn.disabled = true;
  addFriendSendBtn.textContent = 'Sending…';
  addFriendError.textContent = '';
  try {
    const q = query(collection(db, 'userDirectory'), where('email', '==', email));
    const results = await getDocs(q);
    if (results.empty) {
      addFriendError.textContent = "We couldn't find anyone with that email.";
      return;
    }
    const otherUid = results.docs[0].id;
    await sendFriendRequestToUid(otherUid, { fromDisplayName: myFormattedDisplayName() });
    addFriendModalOverlay.classList.remove('show');
    await loadFriendsAndRequests();
  } catch (err) {
    console.error('Add friend failed:', err);
    addFriendError.textContent = err.message || 'Could not send request.';
  } finally {
    addFriendSendBtn.disabled = false;
    addFriendSendBtn.textContent = 'Send Request';
  }
});

// ---- My QR Code — shows this account's uid as a scannable code, so a
// friend can add you (or share a course to you, see learn.js) without
// either of you typing an email. Plain uid payload, same non-URL pattern
// multiplayer.js already uses for game codes: nothing for the system
// Camera app to open, so it only means anything inside Kids Learning Lab's
// own scanner. ----
// While this modal is open, someone else may scan the code on screen at
// any moment — either to add as a friend, or (from Learn's Share Course
// screen) to send a course. Poll every 2s so both show up live instead of
// requiring the learner to back out and reopen Friends to see them.
let myQrPollTimer = null;
let myQrKnownSharedCourseCount = null; // baseline captured when the modal opens

async function pendingSharedCourseCount() {
  const u = auth.currentUser;
  if (!u) return 0;
  try {
    const snap = await getDocs(query(collection(db, 'users', u.uid, 'sharedCourses'), where('status', '==', 'pending')));
    return snap.size;
  } catch {
    return myQrKnownSharedCourseCount ?? 0; // leave the count as-is on a transient read failure
  }
}

function stopMyQrPolling() {
  if (myQrPollTimer) clearInterval(myQrPollTimer);
  myQrPollTimer = null;
}

myQrOpenBtn?.addEventListener('click', async () => {
  const u = auth.currentUser;
  if (!u || !myQrCanvas) return;
  renderJoinQr(myQrCanvas, u.uid).catch((err) => console.error('Could not render QR code:', err));
  myQrModalOverlay.classList.add('show');

  myQrKnownSharedCourseCount = await pendingSharedCourseCount();
  stopMyQrPolling();
  myQrPollTimer = setInterval(async () => {
    if (!myQrModalOverlay.classList.contains('show')) { stopMyQrPolling(); return; }
    loadFriendsAndRequests();

    const count = await pendingSharedCourseCount();
    if (myQrKnownSharedCourseCount != null && count > myQrKnownSharedCourseCount) {
      myQrKnownSharedCourseCount = count;
      stopMyQrPolling();
      showCourseDetectedThenOpenLearn();
    } else {
      myQrKnownSharedCourseCount = count;
    }
  }, 2000);
});
myQrCloseBtn?.addEventListener('click', () => {
  myQrModalOverlay.classList.remove('show');
  stopMyQrPolling();
});

// Small "aesthetic" beat once a shared course is detected in the background:
// a brief status line + fake progress bar, then hop straight to Learn where
// the shared-course banner (see learn.js) picks it up for real.
function showCourseDetectedThenOpenLearn() {
  if (myQrCourseDetectedEl) {
    myQrCourseDetectedEl.style.display = '';
    const fill = myQrCourseDetectedEl.querySelector('.my-qr-detect-fill');
    if (fill) {
      fill.style.transition = 'none';
      fill.style.width = '0%';
      requestAnimationFrame(() => {
        fill.style.transition = 'width 3s linear';
        fill.style.width = '100%';
      });
    }
  }
  setTimeout(async () => {
    if (myQrCourseDetectedEl) myQrCourseDetectedEl.style.display = 'none';
    myQrModalOverlay.classList.remove('show');
    friendsPageOverlay.classList.remove('show');
    await ensureLearnInitialized();
    goToLearnTab();
  }, 3000);
}

// ---- Scan to Add Friend — camera scans a friend's "My QR Code" (their raw
// uid) and runs it through the exact same request logic as the email flow.
// Routed through identifyScannedCode() rather than assumed to be a person
// code — if this actually turns out to be a game join code (scanned by
// mistake, or just handed the wrong QR), it opens that game instead. Uid
// payloads are mixed-case, so this MUST scan with preserveCase: true —
// unlike the 4-char game codes, uppercasing would break the lookup. ----
scanAddFriendBtn?.addEventListener('click', async () => {
  scanAddFriendBtn.disabled = true;
  try {
    const scanned = await scanJoinCode({
      statusText: "Point the camera at your friend's QR code",
      preserveCase: true,
    });
    if (!scanned) return; // user canceled

    await maybeShowOverlay('qrProcessing');
    const result = await identifyScannedCode(scanned);

    if (result.type === 'game') {
      // Not a friend code — a live game session code. Jump into it.
      friendsPageOverlay.classList.remove('show');
      try {
        await joinGameByCode(result.code);
      } catch (err) {
        friendsPageOverlay.classList.add('show');
        addFriendEmailInput.value = '';
        addFriendError.textContent = err.message || "Couldn't join that game.";
        addFriendModalOverlay.classList.add('show');
      }
      return;
    }

    if (result.type === 'unknown') {
      addFriendEmailInput.value = '';
      addFriendError.textContent = "That code wasn't recognized.";
      addFriendModalOverlay.classList.add('show');
      return;
    }

    await sendFriendRequestToUid(result.uid, { fromDisplayName: myFormattedDisplayName() });
    await loadFriendsAndRequests();
  } catch (err) {
    // Surface the error in the Add Friend modal (open it if it wasn't
    // already) so the person sees why the scan didn't work.
    addFriendEmailInput.value = '';
    addFriendError.textContent = err.message || 'Could not read that code.';
    addFriendModalOverlay.classList.add('show');
  } finally {
    scanAddFriendBtn.disabled = false;
  }
});

async function respondToFriendRequest(otherUid, accept) {
  const u = auth.currentUser;
  if (!u) return;
  try {
    if (accept) {
      await setDoc(doc(db, 'users', u.uid, 'friends', otherUid), { status: 'accepted' }, { merge: true });
      await setDoc(doc(db, 'users', otherUid, 'friends', u.uid), { status: 'accepted' }, { merge: true });
      notifyUser(otherUid, {
        type: 'friend_accepted',
        title: 'Friend request accepted',
        body: `${formatDisplayName(u.displayName || (u.email ? u.email.split('@')[0] : 'Someone'), isPremium())} accepted your friend request`,
        data: { fromUid: u.uid },
      });
    } else {
      await deleteDoc(doc(db, 'users', u.uid, 'friends', otherUid));
      await deleteDoc(doc(db, 'users', otherUid, 'friends', u.uid));
    }
    await loadFriendsAndRequests();
  } catch (err) {
    console.error('Failed to respond to friend request:', err);
  }
}

// Lets the sender of a still-pending request cancel it — same underlying
// effect as a decline, just triggered from the other side.
async function cancelFriendRequest(otherUid) {
  const u = auth.currentUser;
  if (!u) return;
  try {
    await deleteDoc(doc(db, 'users', u.uid, 'friends', otherUid));
    await deleteDoc(doc(db, 'users', otherUid, 'friends', u.uid));
    await loadFriendsAndRequests();
  } catch (err) {
    console.error('Failed to cancel friend request:', err);
  }
}

async function removeFriend(otherUid, alsoCloseProfile) {
  const u = auth.currentUser;
  if (!u) return;
  try {
    await deleteDoc(doc(db, 'users', u.uid, 'friends', otherUid));
    await deleteDoc(doc(db, 'users', otherUid, 'friends', u.uid));
    if (alsoCloseProfile) friendProfilePageOverlay.classList.remove('show');
    await loadFriendsAndRequests();
  } catch (err) {
    console.error('Failed to remove friend:', err);
  }
}

// ---- Block ----
blockOpenBtn.addEventListener('click', () => {
  blockEmailInput.value = '';
  blockError.textContent = '';
  blockModalOverlay.classList.add('show');
});
blockCancelBtn.addEventListener('click', () => blockModalOverlay.classList.remove('show'));

blockConfirmBtn.addEventListener('click', async () => {
  const email = blockEmailInput.value.trim().toLowerCase();
  const u = auth.currentUser;
  if (!u) return;
  if (!email || !email.includes('@')) { blockError.textContent = 'Enter a valid email address.'; return; }

  blockConfirmBtn.disabled = true;
  blockConfirmBtn.textContent = 'Blocking…';
  blockError.textContent = '';
  try {
    const q = query(collection(db, 'userDirectory'), where('email', '==', email));
    const results = await getDocs(q);
    if (results.empty) {
      blockError.textContent = "We couldn't find anyone with that email.";
      return;
    }
    const otherUid = results.docs[0].id;
    await setDoc(doc(db, 'users', u.uid, 'blocked', otherUid), { createdAt: Date.now(), email });
    // Blocking also tears down any existing/pending friendship with them.
    await deleteDoc(doc(db, 'users', u.uid, 'friends', otherUid)).catch(() => {});
    await deleteDoc(doc(db, 'users', otherUid, 'friends', u.uid)).catch(() => {});

    blockModalOverlay.classList.remove('show');
    await loadFriendsAndRequests();
  } catch (err) {
    console.error('Block failed:', err);
    blockError.textContent = err.message || 'Could not block that user.';
  } finally {
    blockConfirmBtn.disabled = false;
    blockConfirmBtn.textContent = 'Block';
  }
});

// ---- Friend's profile (opened by tapping a friend in the list) ----
async function openFriendProfile(otherUid) {
  openFriendProfileUid = otherUid;
  friendProfileTitle.textContent = 'Friend';
  renderAvatarInto(friendProfileAvatar, null);
  friendProfileName.textContent = '—';
  friendProfileEmail.textContent = '—';
  friendProfileXp.textContent = '0';
  friendProfileStreak.textContent = '0';
  friendProfileRemoveBtn.style.display = 'none';
  friendProfileBadges.innerHTML = `<p class="badges-page-sub">Loading…</p>`;
  friendProfileFriendsList.innerHTML = '';
  friendProfileFriendsEmpty.style.display = 'none';
  friendProfilePageOverlay.classList.add('show');

  try {
    const snap = await getDoc(doc(db, 'userProfiles', otherUid));
    if (snap.exists()) {
      const info = snap.data();
      friendProfileTitle.textContent = formatDisplayName(info.displayName || 'Friend', !!info.premium);
      friendProfileName.textContent = formatDisplayName(info.displayName || info.email || 'Learner', !!info.premium);
      friendProfileEmail.textContent = info.email || '';
      friendProfileXp.textContent = info.xp || 0;
      friendProfileStreak.textContent = info.streak || 0;
      try {
        const avatarSnap = await rtdbGet(rtdbRef(rtdb, `avatars/${otherUid}`));
        if (avatarSnap.exists()) {
          const avatarData = avatarSnap.val();
          renderAvatarInto(friendProfileAvatar, { emoji: avatarData.emoji, color: avatarData.color });
        }
      } catch { /* not readable / no avatar set — leave placeholder */ }
      friendProfileBadges.innerHTML = renderBadgeGridHtml(info.badges || {});
    } else {
      friendProfileBadges.innerHTML = `<p class="badges-page-sub">No badges earned yet.</p>`;
    }
  } catch (err) {
    console.error('Failed to load friend profile:', err);
    friendProfileBadges.innerHTML = `<p class="badges-page-sub">Could not load badges right now.</p>`;
  }

  // Only show "Remove Friend" if we're actually accepted friends with them
  // (as opposed to viewing a friend-of-a-friend from their friends list).
  const u = auth.currentUser;
  if (u) {
    try {
      const relSnap = await getDoc(doc(db, 'users', u.uid, 'friends', otherUid));
      friendProfileRemoveBtn.style.display = (relSnap.exists() && relSnap.data().status === 'accepted') ? '' : 'none';
    } catch { /* leave hidden */ }
  }

  // This person's own friends list — readable under the Firestore rules
  // when we're an accepted friend of theirs.
  try {
    const friendsSnap = await getDocs(collection(db, 'users', otherUid, 'friends'));
    const theirFriends = [];
    friendsSnap.forEach((d) => {
      const data = d.data();
      if (data.status === 'accepted') theirFriends.push({ uid: d.id, ...data });
    });
    if (!theirFriends.length) {
      friendProfileFriendsEmpty.style.display = '';
    } else {
      for (const friend of theirFriends) {
        const info = await fetchMiniProfile(friend.uid);
        const row = document.createElement('div');
        row.className = 'friend-row';
        row.innerHTML = `
          ${miniAvatarHtml(info)}
          <button type="button" class="friend-row-tap" data-uid="${friend.uid}">
            <div class="friend-row-name">${escapeHtmlMain(formatDisplayName(info.displayName || info.email || 'Learner', !!info.premium))}</div>
            <div class="friend-row-sub">${info.xp || 0} XP · ${info.streak || 0} day streak</div>
          </button>
        `;
        row.querySelector('.friend-row-tap').addEventListener('click', () => openFriendProfile(friend.uid));
        friendProfileFriendsList.appendChild(row);
      }
    }
  } catch (err) {
    // Most likely we're not an accepted friend of theirs (2nd-degree view) —
    // fail quietly and just show nothing rather than an error.
    friendProfileFriendsEmpty.style.display = '';
  }
}

friendProfileExitBtn.addEventListener('click', () => friendProfilePageOverlay.classList.remove('show'));
friendProfileRemoveBtn.addEventListener('click', () => {
  if (openFriendProfileUid) removeFriend(openFriendProfileUid, true);
});

document.getElementById('homeGamesViewAllBtn')?.addEventListener('click', () => {
  document.getElementById('gamesPageOverlay').classList.add('show');
});