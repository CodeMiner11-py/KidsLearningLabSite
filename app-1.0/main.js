import { signIn, signUp, resetPassword, logout } from './auth.js';
import { onAuthStateChanged } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";
import { auth, db } from './firebase.js';
import {
  doc, getDoc, setDoc, deleteDoc, collection, query, where, orderBy, limit, getDocs, onSnapshot, updateDoc
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";
import { getMessaging, getToken, onMessage, isSupported as isMessagingSupported } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-messaging.js";
import { updateProfile } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";
// ---- Elements: auth screen ----
import {updatePassword } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";
import { signInWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";
import { resetLearnState, ensureLearnInitialized } from './learn.js';
import { notifyUser } from './notifications.js';
import { initBadgesForUser, resetBadgesState, checkAccountBadge, checkFriendBadges } from './badges.js';

const authScreen = document.getElementById('auth-screen');
const appShell = document.getElementById('app-shell');

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

const splashScreen = document.getElementById('splash-screen');
setTimeout(() => {
  splashScreen.style.display = 'none';
}, 3000);


document.addEventListener('pointerdown', (e) => {
  if (e.target.closest('button, .tab, .nav-btn')) {

    Haptics?.impact({ style: 'HEAVY' });

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
      ? 'Please sign out and sign back in, then try deleting your account again.'
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
  settingsName.textContent = name;
  settingsEmail.textContent = user.email || '';
  renderAvatarInto(settingsAvatar, currentAvatar);
  syncPublicProfileDocs(user);
  loadXpTotal();
}

settingsSignOutBtn.addEventListener('click', async () => {
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

let verifyCode = null;
let verifyExpiry = null;
let verifyTimerInterval = null;
let verifyEmail = null;
let verificationInProgress = false;

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
  usernameSubmitBtn.disabled = true;
  usernameSubmitBtn.textContent = isChange ? 'Saving…' : 'Saving…';
  try {
    await updateProfile(auth.currentUser, { displayName: val });
    await setDoc(doc(db, 'userProfiles', auth.currentUser.uid), { usernameSet: true, displayName: val }, { merge: true });
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
    const result = await signUp(email, password);
    submitBtn.disabled = false;
    submitBtn.textContent = 'Create Account';
    if (!result.success) {
      verificationInProgress = false;
      showMessage(result.error, 'error');
      return;
    }
    await showVerifyModal(email);
  } else {
    const result = await signIn(email, password);
    submitBtn.disabled = false;
    submitBtn.textContent = 'Sign In';
    if (!result.success) showMessage(result.error, 'error');
  }
});

// ---- Auth state -> screen swap ----
let lastAuthUid = null; // tracks whose data is currently loaded, so we can
                         // clear stale in-memory state when the account changes
onAuthStateChanged(auth, async (user) => {
  if (user?.uid !== lastAuthUid) {
    // Different account (or signed out) — wipe any cached data from the
    // previous account before loading/showing the new one. This includes
    // any full-page overlay/modal that might still be marked "show" (e.g.
    // Account Settings, opened from within itself via Sign Out) and the
    // in-memory profile photo, which otherwise keeps showing the previous
    // account's picture until the app is fully restarted.
    resetLearnState();
    resetBadgesState();
    stopXpListener();
    stopNotifBell();
    document.querySelectorAll('.kll-modal-overlay.show').forEach((el) => el.classList.remove('show'));
    currentAvatar = null;
    renderAvatarInto(settingsAvatar, null);
    lastAuthUid = user?.uid || null;
  }

  if (user) {
    if (verificationInProgress) return;

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
    updateSettingsUI(user);   // ← make sure this line is here
    await initBadgesForUser(user.uid);
    checkAccountBadge(true); // usernameSet (checked above) implies email verification + username both done
    startXpListener(user);
    startNotifBell(user);
    initPushForCurrentUser(); // ask for push permission on first app open, not just on first bell tap
    refreshHome();
  } else {
    authScreen.style.display = 'flex';
    appShell.style.display = 'none';
  }
});

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

navButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    navButtons.forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');

    const target = btn.dataset.page;
    pages.forEach((p) => {
      p.style.display = p.id === `page-${target}` ? 'flex' : 'none';
    });
    updateNotifBellVisibility(target);

    if (target === 'home') refreshHome();
  });
});

// Set correct initial visibility for whichever nav button is active on
// first load (currently "home", where the bell should be hidden).
updateNotifBellVisibility(document.querySelector('.nav-btn.active')?.dataset.page);

// ---- Settings: sign out (temporary, until settings page is built out) ----
const signOutBtn = document.getElementById('signout-btn');
if (signOutBtn) {
  signOutBtn.addEventListener('click', async () => {
    await logout();
  });
}
// ============================================================
// HOME PAGE — greeting + randomized, colorful widget grid
// ============================================================
const homeGreeting = document.getElementById('homeGreeting');
const homeWidgetsEl = document.getElementById('homeWidgets');

let homeLoadToken = 0; // bumped on every refresh so a slow, stale fetch can't clobber a newer render

// Every playable game, with the exact card ID to trigger from Games so a
// widget tap can jump straight into one instead of the games list.
const HOME_GAME_LIST = [
  { name: 'Trivia', desc: 'Quick-fire questions about anything', cardId: 'gameCardTrivia' },
  { name: 'Maze', desc: 'Navigate a maze, answer questions to keep moving', cardId: 'gameCardMaze' },
  { name: 'Seesaw', desc: '2 players, pass the phone — answer before time runs out', cardId: 'gameCardSeesaw' },
  { name: 'Meltdown', desc: 'Solo speed round — answer fast before you melt', cardId: 'gameCardMeltdown' },
  { name: 'Who Can Answer First?', desc: '2 players, same question — fastest correct tap wins', cardId: 'gameCardDuel' },
];

// Color pairs (gradient start/end) — one fixed identity per widget "slot"
// so the grid stays visually consistent between refreshes, iOS-widget style.
const HOME_COLORS = {
  streak: ['#FF8A2B', '#FF6A1F'],
  course: ['#FFCB3D', '#FFB020'],
  game: ['#5FCB58', '#3DAE45'],
  wrong: ['#FF6B7A', '#F0435A'],
  episode: ['#29C2D8', '#1AA0C2'],
  friends: ['#9B6BFF', '#7C4DE0'],
  shared: ['#FF6FA8', '#E8508C'],
  profile: ['#6C7CE0', '#4F5BC7'],
};

// Non-emoji icon for each widget "slot" — streak uses the streak.png image
// file (sits next to index.html, not in an assets folder); everything else
// uses the same Material Symbols font already used across the rest of the
// app, so nothing on Home renders as an emoji character.
const HOME_ICONS = {
  course: 'celebration',
  wrong: 'history_edu',
  episode: 'podcasts',
  friends: 'group_add',
  shared: 'card_giftcard',
  profile: 'edit',
};

function todayStrHome() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

// Jumps to the Learn tab exactly like tapping the nav button would (page
// swap + Learn's own lazy init), so widgets can send the learner there.
function goToLearnTab() {
  document.querySelector('.nav-btn[data-page="learn"]')?.click();
}

async function refreshHome() {
  const u = auth.currentUser;
  if (!u || !homeWidgetsEl) return;
  const token = ++homeLoadToken;

  const name = u.displayName || (u.email ? u.email.split('@')[0] : 'Learner');
  if (homeGreeting) homeGreeting.textContent = `Welcome, ${name}!`;

  const data = await loadHomeData(u.uid);
  if (token !== homeLoadToken) return; // a newer refresh already started
  renderHomeWidgets(buildHomeWidgets(data));
}

// Pulls everything Home's widgets might need directly from Firestore (and,
// for the latest episode, from the Listen tab's own already-rendered DOM),
// independent of whether the learner has opened Learn/Listen/Friends yet.
async function loadHomeData(uidStr) {
  const [learnProfileSnap, coursesSnap, friendsSnap, sharedSnap] = await Promise.all([
    getDoc(doc(db, 'users', uidStr, 'learnProfile', 'main')).catch(() => null),
    getDocs(query(collection(db, 'users', uidStr, 'learnCourses'), orderBy('lastOpenedAt', 'desc'), limit(1))).catch(() => null),
    getDocs(collection(db, 'users', uidStr, 'friends')).catch(() => null),
    getDocs(query(collection(db, 'users', uidStr, 'sharedCourses'), where('status', '==', 'pending'))).catch(() => null),
  ]);

  const learnProfile = learnProfileSnap?.exists()
    ? learnProfileSnap.data()
    : { streak: 0, xp: 0, lastLessonDate: null, missedDaysInRow: 0 };

  // Same "more than 2 full days since last lesson -> streak's actually 0"
  // decay Learn applies, computed read-only here just for messaging.
  let effectiveStreak = learnProfile.streak || 0;
  let missedDaysInRow = 0;
  if (learnProfile.lastLessonDate) {
    const last = new Date(learnProfile.lastLessonDate + 'T00:00:00');
    const today = new Date(todayStrHome() + 'T00:00:00');
    const daysSince = Math.round((today - last) / 86400000);
    missedDaysInRow = Math.max(0, daysSince - 1);
    if (missedDaysInRow > 2) effectiveStreak = 0;
  }
  const doneToday = learnProfile.lastLessonDate === todayStrHome();

  const activeCourse = coursesSnap && !coursesSnap.empty
    ? { id: coursesSnap.docs[0].id, ...coursesSnap.docs[0].data() }
    : null;

  let wrongCount = 0;
  if (activeCourse) {
    const wrongSnap = await getDocs(collection(db, 'users', uidStr, 'learnCourses', activeCourse.id, 'wrongAnswers')).catch(() => null);
    wrongCount = wrongSnap ? wrongSnap.size : 0;
  }

  let friendReqCount = 0;
  if (friendsSnap) {
    friendsSnap.forEach((d) => {
      const fd = d.data();
      if (fd.status === 'pending' && fd.direction === 'received') friendReqCount++;
    });
  }

  const sharedCourseCount = sharedSnap ? sharedSnap.size : 0;

  // Latest episode: read straight off the Listen tab's own DOM once it's
  // rendered its list — avoids duplicating player.js's fetch/parse logic.
  let latestEpisode = null;
  const firstCard = document.querySelector('#listenEpisodeList .episode-card');
  if (firstCard) {
    latestEpisode = {
      title: firstCard.querySelector('.episode-card-title')?.textContent?.trim() || null,
    };
  } else {
    // Episodes usually aren't loaded yet the first time Home renders (right
    // after sign-in) — watch for them and re-render Home once, if the
    // learner is still sitting on the Home tab when they show up.
    watchForLatestEpisode();
  }

  return { learnProfile, effectiveStreak, missedDaysInRow, doneToday, activeCourse, wrongCount, friendReqCount, sharedCourseCount, latestEpisode };
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

// Builds the pool of eligible widgets from fetched data — each one carries
// its own "why am I showing this" calculation, so the copy always reflects
// the learner's actual state, plus the size/color/template it should
// render as (small/wide/tall/large/banner — iOS-widget style).
function buildHomeWidgets(data) {
  const { effectiveStreak, missedDaysInRow, doneToday, activeCourse, wrongCount, friendReqCount, sharedCourseCount, latestEpisode } = data;
  const widgets = [];

  // ---- Streak — small "stat" widget: big number + flame ----
  const [streakA, streakB] = HOME_COLORS.streak;
  if (effectiveStreak === 0) {
    widgets.push({
      key: 'streak', size: 'sm', template: 'stat', colors: [streakA, streakB],
      num: '0', label: 'start today',
      onClick: goToLearnTab,
    });
  } else {
    widgets.push({
      key: 'streak', size: 'sm', template: 'stat', colors: [streakA, streakB],
      num: String(effectiveStreak),
      label: doneToday ? 'day streak, done!' : (missedDaysInRow >= 1 ? 'day streak — don\'t lose it!' : 'day streak'),
      onClick: doneToday
        ? async () => { await ensureLearnInitialized(); goToLearnTab(); document.getElementById('learnStreakBtn')?.click(); }
        : goToLearnTab,
    });
  }

  // ---- Edit profile — tiny "icon" widget, exactly 1 row tall so it
  // slots into the gap the grid otherwise leaves under the streak card
  // (2 rows) when it lands beside the taller game card (3 rows) ----
  const [profA, profB] = HOME_COLORS.profile;
  widgets.push({
    key: 'editProfile', size: 'xs', template: 'icon', colors: [profA, profB], icon: HOME_ICONS.profile,
    label: 'Edit your profile',
    onClick: () => document.querySelector('.nav-btn[data-page="settings"]')?.click(),
  });

  // ---- Continue course / start one / review / complete — wide "info" ----
  const [courseA, courseB] = HOME_COLORS.course;
  if (!activeCourse) {
    widgets.push({
      key: 'course', size: 'wide', template: 'info', colors: [courseA, courseB],
      label: 'Start Learning', big: 'New Course', sub: 'Type anything you want to learn',
      onClick: goToLearnTab,
    });
  } else {
    const courseDone = (activeCourse.currentUnitIndex || 0) >= 10;
    if (activeCourse.status === 'generating' && !courseDone) {
      widgets.push({
        key: 'course', size: 'wide', template: 'info', colors: [courseA, courseB],
        label: 'Preparing your lesson…', big: activeCourse.title, sub: 'Check back in a moment',
        onClick: goToLearnTab,
      });
    } else if (courseDone && activeCourse.courseReviewCompleted) {
      widgets.push({
        key: 'course', size: 'wide', template: 'info', colors: [courseA, courseB],
        label: 'Course complete!', big: activeCourse.title, sub: 'Ready for something new?',
        onClick: goToLearnTab,
      });
    } else if (courseDone) {
      widgets.push({
        key: 'course', size: 'wide', template: 'info', colors: [courseA, courseB],
        label: 'Final Review Ready', big: activeCourse.title, sub: '15 questions on the whole course',
        onClick: goToLearnTab,
      });
    } else {
      widgets.push({
        key: 'course', size: 'wide', template: 'info', colors: [courseA, courseB],
        label: 'Continue where you left off', big: activeCourse.title,
        sub: `Unit ${(activeCourse.currentUnitIndex || 0) + 1}, Lesson ${(activeCourse.currentLessonIndex || 0) + 1}`,
        onClick: goToLearnTab,
      });
    }
  }

  // ---- Wrong answers — full-width "banner" ----
  if (wrongCount > 0) {
    const [wrongA, wrongB] = HOME_COLORS.wrong;
    widgets.push({
      key: 'wrong', size: 'banner', template: 'banner', colors: [wrongA, wrongB], icon: HOME_ICONS.wrong,
      big: `${wrongCount} Wrong Answer${wrongCount === 1 ? '' : 's'}`,
      onClick: async () => { await ensureLearnInitialized(); goToLearnTab(); },
    });
  }

  // ---- Play a new game — tall "info", one specific game picked at random ----
  const pickedGame = HOME_GAME_LIST[Math.floor(Math.random() * HOME_GAME_LIST.length)];
  const [gameA, gameB] = HOME_COLORS.game;
  widgets.push({
    key: 'game', size: 'tall', template: 'info', colors: [gameA, gameB],
    label: 'Play a new game', big: pickedGame.name, sub: pickedGame.desc,
    onClick: async () => { await ensureLearnInitialized(); document.getElementById(pickedGame.cardId)?.click(); },
  });

  // ---- Latest episode — full-width "banner" ----
  const [epA, epB] = HOME_COLORS.episode;
  widgets.push({
    key: 'episode', size: 'banner', template: 'banner', colors: [epA, epB], icon: HOME_ICONS.episode,
    big: latestEpisode?.title ? latestEpisode.title : 'Catch up on the podcast',
    onClick: () => document.querySelector('.nav-btn[data-page="listen"]')?.click(),
  });

  // ---- Friend requests — full-width "banner" ----
  if (friendReqCount > 0) {
    const [frA, frB] = HOME_COLORS.friends;
    widgets.push({
      key: 'friends', size: 'banner', template: 'banner', colors: [frA, frB], icon: HOME_ICONS.friends,
      big: `${friendReqCount} Friend Request${friendReqCount === 1 ? '' : 's'}`,
      onClick: () => profileFriendsBtn?.click(),
    });
  }

  // ---- Shared courses — full-width "banner" ----
  if (sharedCourseCount > 0) {
    const [shA, shB] = HOME_COLORS.shared;
    widgets.push({
      key: 'shared', size: 'banner', template: 'banner', colors: [shA, shB], icon: HOME_ICONS.shared,
      big: `${sharedCourseCount} Course${sharedCourseCount === 1 ? '' : 's'} Shared With You`,
      onClick: async () => { await ensureLearnInitialized(); goToLearnTab(); },
    });
  }

  return widgets;
}

// Fisher–Yates shuffle.
function shuffleHomeWidgets(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// Every possible widget "slot" — shuffled once into a stable order the
// moment Home first renders after an app load/reload, then reused for every
// later refresh (opening the tab again, data changing, etc.) so the layout
// doesn't reshuffle itself just from navigating back to Home. A fresh
// shuffle only happens again after the app itself is reloaded.
const HOME_WIDGET_KEYS = ['streak', 'course', 'wrong', 'game', 'episode', 'friends', 'shared', 'editProfile'];
let homeWidgetOrder = null;
function getHomeWidgetOrder() {
  if (!homeWidgetOrder) homeWidgetOrder = shuffleHomeWidgets(HOME_WIDGET_KEYS);
  return homeWidgetOrder;
}

const HOME_SIZE_CLASS = { xs: 'hw-xs', sm: 'hw-sm', wide: 'hw-wide', tall: 'hw-tall', large: 'hw-large', banner: 'hw-banner' };

function renderHomeWidgets(widgets) {
  if (!homeWidgetsEl) return;
  homeWidgetsEl.innerHTML = '';
  const order = getHomeWidgetOrder();
  const sorted = [...widgets].sort((a, b) => order.indexOf(a.key) - order.indexOf(b.key));

  sorted.forEach((w) => {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = `home-widget type-${w.template} ${HOME_SIZE_CLASS[w.size] || 'hw-wide'}`;
    card.style.background = `linear-gradient(135deg, ${w.colors[0]}, ${w.colors[1]})`;

    if (w.template === 'stat') {
      card.innerHTML = `
        <div class="hw-stat-num">${escapeHtmlMain(w.num)}<img src="./streak.png" class="hw-stat-icon" alt="" /></div>
        <div class="hw-stat-label">${escapeHtmlMain(w.label)}</div>
      `;
    } else if (w.template === 'banner') {
      card.innerHTML = `
        ${w.icon ? `<span class="material-symbols-outlined hw-banner-icon">${w.icon}</span>` : ''}
        <div class="hw-big">${escapeHtmlMain(w.big)}</div>
      `;
    } else if (w.template === 'icon') {
      card.innerHTML = `
        <span class="material-symbols-outlined hw-icon-symbol">${w.icon}</span>
        <div class="hw-icon-label">${escapeHtmlMain(w.label)}</div>
      `;
    } else {
      card.innerHTML = `
        <div class="hw-label">${escapeHtmlMain(w.label)}</div>
        <div class="hw-big">${escapeHtmlMain(w.big)}</div>
        ${w.sub ? `<div class="hw-sub">${escapeHtmlMain(w.sub)}</div>` : ''}
      `;
    }
    card.addEventListener('click', () => w.onClick && w.onClick());
    homeWidgetsEl.appendChild(card);
  });

  fitHomeGrid();
}

// Sizes the grid's row height on the fly so the whole widget grid — however
// many widgets happen to be showing — always exactly fills the space below
// the greeting with no leftover empty area and no overflow/scrolling.
function fitHomeGrid() {
  if (!homeWidgetsEl) return;
  requestAnimationFrame(() => {
    const available = homeWidgetsEl.clientHeight;
    if (!available) return;

    const rowsTemplate = getComputedStyle(homeWidgetsEl).gridTemplateRows;
    const rowCount = rowsTemplate.split(' ').filter(Boolean).length;
    if (!rowCount) return;

    const gap = 14; // matches .home-widgets-grid's gap
    const rowHeight = (available - gap * (rowCount - 1)) / rowCount;
    homeWidgetsEl.style.setProperty('--hw-row', `${Math.max(44, rowHeight)}px`);
  });
}

window.addEventListener('resize', () => fitHomeGrid());

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

// Build the emoji grid buttons once.
AVATAR_EMOJI_CHOICES.forEach((emoji) => {
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

// Build the color swatch buttons once.
AVATAR_COLOR_CHOICES.forEach((color) => {
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
    await setDoc(doc(db, 'userProfiles', u.uid), {
      avatarEmoji: avatar.emoji,
      avatarColor: avatar.color,
    }, { merge: true });
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
    }, { merge: true });
  } catch (err) {
    console.error('Failed to sync public profile docs:', err);
  }
}

async function loadXpTotal() {
  const u = auth.currentUser;
  if (!u) return;
  try {
    const snap = await getDoc(doc(db, 'users', u.uid, 'learnProfile', 'main'));
    const xp = snap.exists() ? (snap.data().xp || 0) : 0;
    profileXpCount.textContent = xp;
    // also pull the saved avatar, if any, now that we know it exists. Always
    // set (or clear) currentAvatar here — if we only set it when an avatar
    // exists, switching to an account with no avatar would keep showing the
    // previous account's avatar.
    const profSnap = await getDoc(doc(db, 'userProfiles', u.uid));
    const profData = profSnap.exists() ? profSnap.data() : {};
    currentAvatar = profData.avatarEmoji ? { emoji: profData.avatarEmoji, color: profData.avatarColor } : null;
    renderAvatarInto(settingsAvatar, currentAvatar);
  } catch (err) {
    console.error('Failed to load XP total:', err);
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

function startNotifBell(user) {
  stopNotifBell();
  const q = query(collection(db, 'users', user.uid, 'notifications'), orderBy('createdAt', 'desc'), limit(30));
  unsubscribeNotifs = onSnapshot(q, (snap) => {
    const notifs = snap.docs.map((d) => ({ id: d.id, ...d.data() }));
    renderNotifBadge(notifs);
    renderNotifPanel(notifs);
  }, (err) => console.error('Notifications listener failed:', err));
}
function stopNotifBell() {
  if (unsubscribeNotifs) { unsubscribeNotifs(); unsubscribeNotifs = null; }
  notifBellBadge.style.display = 'none';
  notifPanelList.innerHTML = '';
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
const friendProfileFriendsEmpty = document.getElementById('friendProfileFriendsEmpty');
const friendProfileFriendsList = document.getElementById('friendProfileFriendsList');

let openFriendProfileUid = null; // whoever's profile is currently open in the friend-profile fullpager

profileFriendsBtn.addEventListener('click', () => {
  friendsPageOverlay.classList.add('show');
  loadFriendsPage();
});
friendsExitBtn.addEventListener('click', () => friendsPageOverlay.classList.remove('show'));

async function loadFriendsPage() {
  const u = auth.currentUser;
  if (!u) return;

  // ---- My own summary header ----
  renderAvatarInto(friendsMyAvatar, currentAvatar);
  friendsMyName.textContent = u.displayName || (u.email ? u.email.split('@')[0] : 'Learner');
  friendsMyEmail.textContent = u.email || '';
  try {
    const learnSnap = await getDoc(doc(db, 'users', u.uid, 'learnProfile', 'main'));
    const data = learnSnap.exists() ? learnSnap.data() : {};
    friendsMyXp.textContent = data.xp || 0;
    friendsMyStreak.textContent = data.streak || 0;
  } catch { /* leave as-is on failure */ }

  await loadFriendsAndRequests();
}

async function loadFriendsAndRequests() {
  const u = auth.currentUser;
  if (!u) return;
  friendRequestsList.innerHTML = '';
  sentRequestsList.innerHTML = '';
  friendsList.innerHTML = '';
  friendRequestsSection.style.display = 'none';
  sentRequestsSection.style.display = 'none';
  friendsListEmpty.style.display = 'none';

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
          <div class="friend-row-name">${escapeHtmlMain(info.displayName || info.email || 'Learner')}</div>
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
          <div class="friend-row-name">${escapeHtmlMain(info.displayName || info.email || 'Learner')}</div>
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
    friendsListEmpty.style.display = '';
  } else {
    for (const friend of accepted) {
      const info = await fetchMiniProfile(friend.uid);
      const row = document.createElement('div');
      row.className = 'friend-row';
      row.innerHTML = `
        ${miniAvatarHtml(info)}
        <button type="button" class="friend-row-tap" data-uid="${friend.uid}">
          <div class="friend-row-name">${escapeHtmlMain(info.displayName || info.email || 'Learner')}</div>
          <div class="friend-row-sub">${info.xp || 0} XP · ${info.streak || 0} day streak</div>
        </button>
        <button type="button" class="friend-row-btn" data-uid="${friend.uid}">Remove Friend</button>
      `;
      row.querySelector('.friend-row-tap').addEventListener('click', () => openFriendProfile(friend.uid));
      row.querySelector('.friend-row-btn').addEventListener('click', () => removeFriend(friend.uid, false));
      friendsList.appendChild(row);
    }
  }
}

// Reads a friend/candidate's public profile doc. Only succeeds under the
// Firestore rules if we're either the owner or an accepted friend — for a
// pending *incoming* request the sender's profile isn't readable yet, so we
// fall back to userDirectory (email only) in that case.
async function fetchMiniProfile(otherUid) {
  try {
    const snap = await getDoc(doc(db, 'userProfiles', otherUid));
    if (snap.exists()) return snap.data();
  } catch { /* likely not-yet-accepted — fall through */ }
  try {
    const dirSnap = await getDoc(doc(db, 'userDirectory', otherUid));
    if (dirSnap.exists()) return { email: dirSnap.data().email };
  } catch { /* ignore */ }
  return {};
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

    // If they already sent *us* a request, accept it instead of creating a
    // duplicate reverse request.
    const existingReverse = await getDoc(doc(db, 'users', u.uid, 'friends', otherUid));
    if (existingReverse.exists() && existingReverse.data().status === 'pending' && existingReverse.data().direction === 'received') {
      await respondToFriendRequest(otherUid, true);
      addFriendModalOverlay.classList.remove('show');
      return;
    }
    if (existingReverse.exists() && existingReverse.data().status === 'accepted') {
      addFriendError.textContent = "You're already friends!";
      return;
    }

    await setDoc(doc(db, 'users', u.uid, 'friends', otherUid), {
      status: 'pending', direction: 'sent', createdAt: Date.now(),
    });
    await setDoc(doc(db, 'users', otherUid, 'friends', u.uid), {
      status: 'pending', direction: 'received', createdAt: Date.now(),
    });
    notifyUser(otherUid, {
      type: 'friend_request',
      title: 'New friend request',
      body: `${u.displayName || (u.email ? u.email.split('@')[0] : 'Someone')} wants to be friends`,
      data: { fromUid: u.uid },
    });

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
        body: `${u.displayName || (u.email ? u.email.split('@')[0] : 'Someone')} accepted your friend request`,
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
  friendProfileFriendsList.innerHTML = '';
  friendProfileFriendsEmpty.style.display = 'none';
  friendProfilePageOverlay.classList.add('show');

  try {
    const snap = await getDoc(doc(db, 'userProfiles', otherUid));
    if (snap.exists()) {
      const info = snap.data();
      friendProfileTitle.textContent = info.displayName || 'Friend';
      friendProfileName.textContent = info.displayName || info.email || 'Learner';
      friendProfileEmail.textContent = info.email || '';
      friendProfileXp.textContent = info.xp || 0;
      friendProfileStreak.textContent = info.streak || 0;
      if (info.avatarEmoji) renderAvatarInto(friendProfileAvatar, { emoji: info.avatarEmoji, color: info.avatarColor });
    }
  } catch (err) {
    console.error('Failed to load friend profile:', err);
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
            <div class="friend-row-name">${escapeHtmlMain(info.displayName || info.email || 'Learner')}</div>
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