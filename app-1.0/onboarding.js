// onboarding.js — Kids Learning Lab first-run onboarding wizard
//
// A 9-step wizard shown the very first time a NEW device opens the app,
// replacing the plain sign-in/create-account screen for that one run.
// Tapping "Sign in" at Step 1 exits the wizard entirely (permanently, on
// this device) and falls through to the existing auth-screen sign-in flow.
// Everything from "Create account" onward (Steps 2-9) is a new UI on top
// of the existing account-creation primitives (auth.js signUp, the
// userProfiles/avatars data model) — it does not replace or duplicate them.
//
// RESUMABILITY: progress is saved to localStorage after every step so a
// learner who backgrounds/closes the app mid-onboarding picks up where
// they left off next launch, rather than starting over or (worse) ending
// up in a broken half-created state. See saveState()/loadState() below.
// The password itself is NEVER persisted. Once the real Firebase account
// has actually been created (end of Step 6), saved state jumps straight to
// step 7 — there is no scenario where a resumed session re-asks for a
// password against an email that's already registered.
import { auth, db, rtdb } from './firebase.js';
import { signUp } from './auth.js';
import { updateProfile } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";
import { doc, setDoc } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";
import { ref as rtdbRef, set as rtdbSet } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-database.js";

const SEEN_KEY = 'kll_onboarding_seen';       // set once wizard is exited (either path), never shown again on this device
const STATE_KEY = 'kll_onboarding_state';     // resumable in-progress state (never includes password)

// Same rules main.js enforces for the username/Change Name modal — kept in
// sync manually since main.js doesn't export its copies of these.
const USERNAME_ALLOWED = /^[a-zA-Z0-9 ]+$/;
const RESERVED_NAME_PATTERN = /premium|\bpro\b/i;
const EMAIL_ALLOWED = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const AVATAR_EMOJI_CHOICES = [
  '🦁', '🐶', '🐱', '🐸', '🦊', '🐼', '🐵', '🦄', '🐬', '🦋', '🐝', '🐢', '🦖', '🐙', '🐰', '🦉',
  '🚀', '⭐', '🌈', '🔥', '⚡', '🎨', '🎮', '📚', '🎵', '⚽', '🏀', '🎸', '🍕', '🍦',
];
const AVATAR_COLOR_CHOICES = [
  '#FF6B6B', '#FF8A2B', '#FFC93C', '#6BCB77', '#34D8A6',
  '#2BC5C5', '#4FA6FF', '#1E6FE0', '#123B7A', '#7C6BFF',
  '#B36BFF', '#FF6BCB', '#FF3B6E', '#C77D3B', '#8D99AE',
];

const RESEND_WORKER_URL = 'https://emailworkerkidslearninglabanyhtmlnonspecific.nameless-cherry-998c.workers.dev/send';

function generateCode() {
  return String(Math.floor(100000 + Math.random() * 900000));
}

async function sendOnboardingVerificationEmail(email, code) {
  const html = `
    <div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:40px 32px;background:#F5FAFF;border-radius:18px;border:1.5px solid #DCE7F5">
      <img src="https://kidslearninglab.com/wp-content/uploads/2025/02/podcast-logo-app-rounded.png" style="width:48px;height:48px;border-radius:14px;display:block;margin:0 auto 20px">
      <h2 style="text-align:center;color:#14213D;margin-bottom:8px">Verify your email</h2>
      <p style="text-align:center;color:#5B6B85;font-size:14px;line-height:1.6;margin-bottom:28px">Enter this code in Kids Learning Lab to finish creating your account. It expires in 10 minutes.</p>
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

// ---- Resumable state ----
function loadState() {
  try {
    const raw = localStorage.getItem(STATE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}
function saveState(state) {
  try { localStorage.setItem(STATE_KEY, JSON.stringify(state)); } catch { /* best-effort */ }
}
function clearState() {
  try { localStorage.removeItem(STATE_KEY); } catch { /* best-effort */ }
}
function markSeen() {
  try { localStorage.setItem(SEEN_KEY, '1'); } catch { /* best-effort */ }
}
function hasBeenSeen() {
  try { return !!localStorage.getItem(SEEN_KEY); } catch { return false; }
}

// ---- Wizard state (in-memory, hydrated from saved state on init) ----
let state = {
  step: 1,
  choice: null,        // 'signin' | 'create' — Step 1 pick
  displayName: '',
  avatarEmoji: [],
  avatarColor: AVATAR_COLOR_CHOICES[0],
  email: '',
  codeVerified: false,
};

let verifyCode = null;
let verifyExpiry = 0;
let verifyTimerInterval = null;
let resendCooldownInterval = null;

let onExitCallback = null; // called with no args when the wizard is fully dismissed (either path)
let onUpgradeCallback = null; // called if the learner taps Upgrade on Step 9 (opens the paywall)

// ---- DOM refs (queried lazily in init() since index.html markup must exist first) ----
let els = {};

function qs(id) { return document.getElementById(id); }

function cacheEls() {
  els = {
    overlay: qs('onboardingOverlay'),
    progressFill: qs('onboardingProgressFill'),
    stepLabel: qs('onboardingStepLabel'),
    card: qs('onboardingCard'),
    exitBtn: qs('onboardingExitBtn'),
    backBtn: qs('onboardingBackBtn'),

    // Step 1
    step1: qs('onboardingStep1'),
    step1SignIn: qs('onboardingStep1SignIn'),
    step1Create: qs('onboardingStep1Create'),
    step1Select: qs('onboardingStep1Select'),

    // Step 2
    step2: qs('onboardingStep2'),
    nameInput: qs('onboardingNameInput'),
    nameError: qs('onboardingNameError'),
    nameChoose: qs('onboardingNameChoose'),

    // Step 3
    step3: qs('onboardingStep3'),
    avatarPreview: qs('onboardingAvatarPreview'),
    avatarEmojiGrid: qs('onboardingAvatarEmojiGrid'),
    avatarColorGrid: qs('onboardingAvatarColorGrid'),
    avatarError: qs('onboardingAvatarError'),
    avatarChoose: qs('onboardingAvatarChoose'),

    // Step 4
    step4: qs('onboardingStep4'),
    emailInput: qs('onboardingEmailInput'),
    emailError: qs('onboardingEmailError'),
    emailSelect: qs('onboardingEmailSelect'),

    // Step 5
    step5: qs('onboardingStep5'),
    codeInput: qs('onboardingCodeInput'),
    codeSubtitle: qs('onboardingCodeSubtitle'),
    codeExpiry: qs('onboardingCodeExpiry'),
    codeError: qs('onboardingCodeError'),
    codeResend: qs('onboardingCodeResend'),
    codeEnter: qs('onboardingCodeEnter'),

    // Step 6
    step6: qs('onboardingStep6'),
    passwordInput: qs('onboardingPasswordInput'),
    passwordError: qs('onboardingPasswordError'),
    passwordChoose: qs('onboardingPasswordChoose'),

    // Step 7
    step7: qs('onboardingStep7'),
    recapEmail: qs('onboardingRecapEmail'),
    recapContinue: qs('onboardingRecapContinue'),

    // Step 8
    step8: qs('onboardingStep8'),
    diagramsOff: qs('onboardingDiagramsOff'),
    diagramsOn: qs('onboardingDiagramsOn'),

    // Step 9
    step9: qs('onboardingStep9'),
    upgradeBtn: qs('onboardingUpgradeBtn'),
    notNowBtn: qs('onboardingNotNowBtn'),
  };
}

const ALL_STEP_KEYS = ['step1', 'step2', 'step3', 'step4', 'step5', 'step6', 'step7', 'step8', 'step9'];

function renderStep() {
  ALL_STEP_KEYS.forEach((key, idx) => {
    const el = els[key];
    if (!el) return;
    el.style.display = (idx + 1 === state.step) ? 'block' : 'none';
  });

  els.progressFill.style.width = `${(state.step / 9) * 100}%`;
  els.stepLabel.textContent = `Step ${state.step}`;

  // Back only makes sense while nothing irreversible has happened yet —
  // steps 2-6 are all still "before the real account exists" (Step 6's
  // password submit is what actually calls signUp(); if that throws,
  // state.step stays at 6, so Back is exactly what a learner needs to fix
  // a bad email and retry). From Step 7 on, the account is real and nothing
  // before it is safe to revisit, so Back disappears for good from there.
  if (els.backBtn) els.backBtn.style.display = (state.step >= 2 && state.step <= 6) ? 'flex' : 'none';

  // Step 9 flips the card to the tan/green celebratory palette per spec.
  els.card.classList.toggle('onboarding-final-step', state.step === 9);

  if (state.step === 1) {
    els.step1SignIn.classList.toggle('selected', state.choice === 'signin');
    els.step1Create.classList.toggle('selected', state.choice === 'create');
    els.step1Select.disabled = !state.choice;
  }
  if (state.step === 2) {
    els.nameInput.value = state.displayName || '';
  }
  if (state.step === 3) {
    renderAvatarGridsIfNeeded();
    refreshAvatarPreview();
  }
  if (state.step === 4) {
    els.emailInput.value = state.email || '';
  }
  if (state.step === 5) {
    els.codeSubtitle.textContent = `Confirm the code sent to ${state.email}`;
  }
  if (state.step === 7) {
    els.recapEmail.textContent = state.email || '';
  }
}

function goToStep(n, persist = true) {
  state.step = n;
  if (persist) persistProgress();
  renderStep();
}

// Only persists the fields that are safe/useful to resume with — never the
// password, and step 6 itself never gets written (see completeAccountCreation,
// which jumps state.step straight from 6 to 7 once the account is real).
function persistProgress() {
  saveState({
    step: state.step,
    choice: state.choice,
    displayName: state.displayName,
    avatarEmoji: state.avatarEmoji,
    avatarColor: state.avatarColor,
    email: state.email,
    codeVerified: state.codeVerified,
  });
}

// ---- Topbar: exit (X) and back ----
// Bound once per maybeStartOnboarding() call, same as the bindStepN()
// functions below — cheap to rebind since it's just two listeners and this
// only runs once per app session anyway.
function bindTopbar() {
  els.exitBtn?.addEventListener('click', () => {
    // Progress is already auto-persisted after every step (persistProgress,
    // called from goToStep) — exiting here doesn't lose anything except the
    // password field, which is never persisted anyway. Reopening resumes
    // right where they left off, so there's nothing destructive to confirm.
    exitOnboarding();
  });
  els.backBtn?.addEventListener('click', () => {
    if (state.step <= 1) return; // guarded by renderStep's visibility too, but don't trust that alone
    // Going back from Step 6 (whether or not the last attempt just failed)
    // clears any stale error/disabled-button state on the way out, so
    // returning to 6 later starts clean rather than showing an old error.
    if (state.step === 6) { els.passwordError.textContent = ''; }
    goToStep(state.step - 1);
  });
}

// ---- Step 1: sign in vs create account ----
function bindStep1() {
  els.step1SignIn.addEventListener('click', () => {
    state.choice = 'signin';
    renderStep();
  });
  els.step1Create.addEventListener('click', () => {
    state.choice = 'create';
    renderStep();
  });
  els.step1Select.addEventListener('click', () => {
    if (state.choice === 'signin') {
      // Exiting to the existing sign-in UI is a permanent choice for this
      // device, same as finishing the wizard the other way.
      exitOnboarding();
      return;
    }
    goToStep(2);
  });
}

// ---- Step 2: name ----
function bindStep2() {
  els.nameChoose.addEventListener('click', () => {
    const val = els.nameInput.value.trim();
    els.nameError.textContent = '';
    if (!val || val.length < 2) { els.nameError.textContent = 'Name must be at least 2 characters.'; return; }
    if (!USERNAME_ALLOWED.test(val)) { els.nameError.textContent = 'Only letters, numbers, and spaces allowed.'; return; }
    if (RESERVED_NAME_PATTERN.test(val)) { els.nameError.textContent = "That name isn't available."; return; }
    state.displayName = val;
    goToStep(3);
  });
}

// ---- Step 3: avatar ----
let avatarGridsBuilt = false;
function renderAvatarGridsIfNeeded() {
  if (avatarGridsBuilt) return;
  avatarGridsBuilt = true;

  els.avatarEmojiGrid.innerHTML = '';
  AVATAR_EMOJI_CHOICES.forEach((emoji) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'avatar-emoji-btn';
    btn.textContent = emoji;
    btn.dataset.emoji = emoji;
    btn.addEventListener('click', () => {
      const idx = state.avatarEmoji.indexOf(emoji);
      if (idx !== -1) {
        state.avatarEmoji.splice(idx, 1);
      } else {
        if (state.avatarEmoji.length >= 2) state.avatarEmoji.shift();
        state.avatarEmoji.push(emoji);
      }
      els.avatarError.textContent = '';
      refreshAvatarPreview();
    });
    els.avatarEmojiGrid.appendChild(btn);
  });

  els.avatarColorGrid.innerHTML = '';
  AVATAR_COLOR_CHOICES.forEach((color) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'avatar-color-swatch';
    btn.style.background = color;
    btn.dataset.color = color;
    btn.addEventListener('click', () => {
      state.avatarColor = color;
      refreshAvatarPreview();
    });
    els.avatarColorGrid.appendChild(btn);
  });
}

function refreshAvatarPreview() {
  els.avatarEmojiGrid?.querySelectorAll('.avatar-emoji-btn').forEach((btn) => {
    btn.classList.toggle('selected', state.avatarEmoji.includes(btn.dataset.emoji));
  });
  els.avatarColorGrid?.querySelectorAll('.avatar-color-swatch').forEach((btn) => {
    btn.classList.toggle('selected', btn.dataset.color === state.avatarColor);
  });
  if (!els.avatarPreview) return;
  if (state.avatarEmoji.length) {
    els.avatarPreview.style.background = state.avatarColor;
    const joined = state.avatarEmoji.join('');
    const chars = Array.from(joined);
    els.avatarPreview.innerHTML = `<span class="avatar-emoji" style="font-size:${chars.length >= 2 ? '0.62em' : '1em'};">${joined}</span>`;
  } else {
    els.avatarPreview.style.background = '';
    els.avatarPreview.innerHTML = `<span class="material-symbols-outlined">person</span>`;
  }
}

function bindStep3() {
  els.avatarChoose.addEventListener('click', () => {
    if (!state.avatarEmoji.length) {
      els.avatarError.textContent = 'Pick at least 1 emoji.';
      return;
    }
    goToStep(4);
  });
}

// ---- Step 4: email ----
function bindStep4() {
  els.emailSelect.addEventListener('click', async () => {
    const val = els.emailInput.value.trim();
    els.emailError.textContent = '';
    if (!val || !EMAIL_ALLOWED.test(val)) { els.emailError.textContent = "That email address doesn't look right."; return; }
    state.email = val;
    state.codeVerified = false;

    els.emailSelect.disabled = true;
    els.emailSelect.textContent = 'Sending code…';
    try {
      await sendCodeForStep5();
      goToStep(5);
    } catch {
      els.emailError.textContent = 'Could not send code. Try again.';
    } finally {
      els.emailSelect.disabled = false;
      els.emailSelect.textContent = 'Select';
    }
  });
}

// ---- Step 5: confirm code ----
async function sendCodeForStep5() {
  verifyCode = generateCode();
  verifyExpiry = Date.now() + 10 * 60 * 1000;
  await sendOnboardingVerificationEmail(state.email, verifyCode);
  startCodeTimer();
  startResendCooldown();
}

function startCodeTimer() {
  clearInterval(verifyTimerInterval);
  verifyTimerInterval = setInterval(() => {
    const remaining = verifyExpiry - Date.now();
    if (remaining <= 0) {
      clearInterval(verifyTimerInterval);
      els.codeExpiry.textContent = 'Code expired. Please resend.';
      verifyCode = null;
      return;
    }
    const mins = Math.floor(remaining / 60000);
    const secs = Math.floor((remaining % 60000) / 1000);
    els.codeExpiry.textContent = `Code expires in ${mins}:${String(secs).padStart(2, '0')}`;
  }, 1000);
}

function startResendCooldown() {
  clearInterval(resendCooldownInterval);
  let remaining = 30;
  els.codeResend.disabled = true;
  els.codeResend.textContent = `Resend code in ${remaining}`;
  resendCooldownInterval = setInterval(() => {
    remaining -= 1;
    if (remaining <= 0) {
      clearInterval(resendCooldownInterval);
      els.codeResend.disabled = false;
      els.codeResend.textContent = 'Resend code';
      return;
    }
    els.codeResend.textContent = `Resend code in ${remaining}`;
  }, 1000);
}

function bindStep5() {
  els.codeEnter.addEventListener('click', () => {
    const entered = els.codeInput.value.trim();
    els.codeError.textContent = '';
    if (!entered || entered.length < 6) { els.codeError.textContent = 'Please enter the 6-digit code.'; return; }
    if (!verifyCode || Date.now() > verifyExpiry) { els.codeError.textContent = 'Code has expired. Please resend.'; return; }
    if (entered !== verifyCode) {
      els.codeError.textContent = 'Incorrect code. Please try again.';
      els.codeInput.value = '';
      els.codeInput.focus();
      return;
    }
    clearInterval(verifyTimerInterval);
    verifyCode = null;
    state.codeVerified = true;
    persistProgress();
    goToStep(6);
  });

  els.codeResend.addEventListener('click', async () => {
    els.codeError.textContent = '';
    try {
      await sendCodeForStep5();
      els.codeError.style.color = 'var(--blue-main)';
      els.codeError.textContent = 'New code sent!';
      setTimeout(() => { els.codeError.textContent = ''; els.codeError.style.color = 'var(--error)'; }, 3000);
    } catch {
      els.codeError.textContent = 'Could not resend. Please try again.';
    }
  });
}

// ---- Step 6: password -> actually create the account ----
function bindStep6() {
  els.passwordChoose.addEventListener('click', async () => {
    const pw = els.passwordInput.value;
    els.passwordError.textContent = '';
    if (!pw || pw.length < 6) { els.passwordError.textContent = 'Password should be at least 6 characters.'; return; }
    if (!state.codeVerified) {
      // Shouldn't be reachable via normal navigation, but guards against a
      // resumed session that somehow skipped Step 5.
      els.passwordError.textContent = 'Please verify your email first.';
      goToStep(4);
      return;
    }

    els.passwordChoose.disabled = true;
    els.passwordChoose.textContent = 'Creating account…';
    try {
      await completeAccountCreation(pw);
      els.passwordInput.value = ''; // never persisted, and no reason to keep it on screen
      goToStep(7);
    } catch (err) {
      els.passwordError.textContent = err?.message || 'Could not create your account. Try again.';
    } finally {
      els.passwordChoose.disabled = false;
      els.passwordChoose.textContent = 'Choose';
    }
  });
}

// Creates the real Firebase account and writes name/avatar/verified-flag —
// all in the background relative to the wizard's own navigation, per spec
// ("don't continue to the home screen yet"). Throws on failure so the Step
// 6 button handler above can show the error and let the learner retry
// without silently advancing past a failed account creation.
async function completeAccountCreation(password) {
  const result = await signUp(state.email, password, { sendVerification: false });
  if (!result.success) throw new Error(result.error || 'Could not create account.');

  const user = result.user;
  const avatar = { emoji: state.avatarEmoji.join(''), color: state.avatarColor };

  await updateProfile(user, { displayName: state.displayName });
  await setDoc(doc(db, 'userProfiles', user.uid), {
    usernameSet: true,
    displayName: state.displayName,
    emailVerified: true, // our own Step 5 code flow already confirmed this email
    lastLoginAt: Date.now(),
  }, { merge: true });
  await rtdbSet(rtdbRef(rtdb, `avatars/${user.uid}`), avatar);

  // The account now genuinely exists — from here on, a resumed session
  // must never come back to the password field. Advance saved state to 7
  // immediately, before returning control to the caller's goToStep(7).
  state.step = 7;
  persistProgress();
}

// ---- Step 7: recap ----
function bindStep7() {
  els.recapContinue.addEventListener('click', () => goToStep(8));
}

// ---- Step 8: diagrams ----
function bindStep8() {
  els.diagramsOff.addEventListener('click', () => {
    try { localStorage.setItem('kll_diagrams_enabled', '0'); } catch { /* best-effort */ }
    goToStep(9);
  });
  els.diagramsOn.addEventListener('click', () => {
    try { localStorage.setItem('kll_diagrams_enabled', '1'); } catch { /* best-effort */ }
    goToStep(9);
  });
}

// ---- Step 9: upgrade or finish ----
function bindStep9() {
  els.upgradeBtn.addEventListener('click', () => {
    exitOnboarding();
    onUpgradeCallback?.();
  });
  els.notNowBtn.addEventListener('click', () => {
    exitOnboarding();
  });
}

function exitOnboarding() {
  clearInterval(verifyTimerInterval);
  clearInterval(resendCooldownInterval);
  markSeen();
  clearState();
  els.overlay.classList.remove('show');
  onExitCallback?.();
}

// ---- Public API ----

// Returns true if the wizard was actually shown (so main.js knows NOT to
// also render the plain auth screen underneath it). Returns false if this
// device has already been through onboarding once (either exit path).
//
// opts.onExit()    — called once the wizard is dismissed, however it exits
// opts.onUpgrade() — called if the learner chooses to upgrade at Step 9
//                    (main.js is expected to open the paywall from this)
export function maybeStartOnboarding(opts = {}) {
  onExitCallback = opts.onExit || null;
  onUpgradeCallback = opts.onUpgrade || null;

  cacheEls();
  if (!els.overlay) {
    console.warn('Onboarding markup missing from index.html (#onboardingOverlay).');
    return false;
  }

  const saved = loadState();
  if (saved) {
    state = {
      step: saved.step || 1,
      choice: saved.choice || null,
      displayName: saved.displayName || '',
      avatarEmoji: saved.avatarEmoji || [],
      avatarColor: saved.avatarColor || AVATAR_COLOR_CHOICES[0],
      email: saved.email || '',
      codeVerified: !!saved.codeVerified,
    };
    // A resumed session that had reached the password step without the
    // account existing yet (app was killed mid-Step-6) can't safely
    // resume there — password isn't recoverable and re-submitting could
    // look like a duplicate signup. Bounce back to Step 6 is fine UNLESS
    // signUp() already ran, in which case completeAccountCreation() itself
    // advanced saved state to 7 before this could ever be loaded. So the
    // only remaining case is state.step === 6 with no account yet, which
    // is safe to simply resume as-is (password field just re-empty).
  }

  bindTopbar();
  bindStep1(); bindStep2(); bindStep3(); bindStep4();
  bindStep5(); bindStep6(); bindStep7(); bindStep8(); bindStep9();

  els.overlay.classList.add('show');
  renderStep();
  return true;
}