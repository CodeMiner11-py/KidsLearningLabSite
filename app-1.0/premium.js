// premium.js — Kids Learning Lab Premium
//
// Single source of truth in the app for "is this user premium?" and the
// free-vs-premium limits. Firestore (users/{uid}/learnProfile/main.premium)
// is the real source of truth on the backend — it's kept in sync by the
// RevenueCat webhook -> Cloudflare Worker (see revenuecat-webhook-worker/),
// so this stays correct across devices and reinstalls without this module
// ever needing to trust the device alone.
import { auth, db } from './firebase.js';
import {
  doc, getDoc, setDoc, onSnapshot
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";

// ---- RevenueCat config ----
// Public iOS SDK key — from RevenueCat dashboard > Apps > (your iOS app) >
// API keys. Safe to embed client-side (it's a "public" key, can only make
// non-destructive changes).
const RC_API_KEY_IOS = 'appl_zMumAzCKNbvURQKDBZrbrfZBxht';
const ENTITLEMENT_ID = 'premium'; // must match the Entitlement identifier in RevenueCat

// ---- Device lock (one purchase per physical device) ----
// Requires `npm install @capacitor/device && npx cap sync` if not already
// installed — this module assumes the plugin is present natively.
let _deviceIdCache = null;

// Stable per-device identifier. On iOS this is identifierForVendor (IDFV):
// stays the same across reinstalls as long as at least one app from this
// vendor remains on the device, resets only on a full uninstall of all of
// them. Falls back to a generated+localStorage-cached id on web/dev preview
// where the native plugin isn't available (not a real security boundary
// there, but keeps local dev from crashing).
export async function getDeviceId() {
  if (_deviceIdCache) return _deviceIdCache;

  const Device = window.Capacitor?.Plugins?.Device;
  if (Device) {
    try {
      const { identifier } = await Device.getId();
      if (identifier) {
        _deviceIdCache = identifier;
        return _deviceIdCache;
      }
    } catch (err) {
      console.warn('Device.getId() failed, falling back to local id:', err);
    }
  }

  const key = 'kll_fallback_device_id';
  let fallback = localStorage.getItem(key);
  if (!fallback) {
    fallback = 'web-' + crypto.randomUUID();
    localStorage.setItem(key, fallback);
  }
  _deviceIdCache = fallback;
  return _deviceIdCache;
}

function maskEmail(email) {
  if (!email || !email.includes('@')) return 'another account';
  const [name, domain] = email.split('@');
  const visible = name.slice(0, 2);
  return `${visible}${'•'.repeat(Math.max(name.length - 2, 3))}@${domain}`;
}

// Checks whether this device has already been used to purchase Premium on
// a DIFFERENT account than the one currently signed in. Returns null if
// this device is unclaimed or claimed by the current uid (i.e. purchase
// flow is allowed to proceed). Returns a masked email string if blocked.
//
// This is a LIVE, uncached read — call it right before anything that
// actually spends money (see runPurchase() in paywall.js). For opening the
// paywall UI itself, use getCachedDeviceLock() instead: that path doesn't
// gate a purchase, so it's fine (and much faster) to show slightly-stale
// data there while this live check stays the actual gate.
export async function checkDeviceLock() {
  const u = auth.currentUser;
  if (!u) return null;

  try {
    const deviceId = await getDeviceId();
    const snap = await getDoc(doc(db, 'deviceRegistry', deviceId));
    if (!snap.exists()) return null;

    const data = snap.data();
    if (!data.purchasedByUid || data.purchasedByUid === u.uid) return null;

    return maskEmail(data.purchasedByEmail);
  } catch (err) {
    // Fail OPEN, not closed — a Firestore/rules/network hiccup here should
    // never be the reason the paywall silently refuses to open. Worst case
    // of failing open is the device-lock check gets skipped once; worst
    // case of failing closed is Premium becomes unbuyable for everyone.
    console.warn('checkDeviceLock() failed, allowing purchase flow to proceed:', err);
    return null;
  }
}

// ---- Background-refreshed cache of the device-lock check ----
// Exists ONLY to make openPaywall() feel instant instead of blocking on a
// network read every time. It is NOT a substitute for checkDeviceLock() at
// the moment of an actual purchase — runPurchase() in paywall.js must keep
// calling the live checkDeviceLock() right before charging, since that's
// the one check standing between a second account and a free entitlement.
// A stale cache here just means the paywall UI briefly shows the wrong
// state (e.g. offers to sell Premium to an already-locked device for up to
// ~30s) — annoying, not exploitable, since the real gate is still live.
const DEVICE_LOCK_POLL_MS = 30_000;
let _cachedDeviceLock = null;    // last known result: null | maskedEmail string
let _deviceLockPollTimer = null;
let _deviceLockInFlight = null;  // dedupe overlapping poll ticks

async function _pollDeviceLockOnce() {
  if (_deviceLockInFlight) return _deviceLockInFlight;
  _deviceLockInFlight = checkDeviceLock()
    .then((result) => { _cachedDeviceLock = result; return result; })
    .finally(() => { _deviceLockInFlight = null; });
  return _deviceLockInFlight;
}

// Call once right after sign-in (mirrors initPremiumForCurrentUser). Runs
// an immediate check so the cache isn't empty on first paywall open, then
// keeps it fresh every 30s in the background for as long as this device
// has a user signed in.
export function startDeviceLockPolling() {
  stopDeviceLockPolling();
  _pollDeviceLockOnce();
  _deviceLockPollTimer = setInterval(_pollDeviceLockOnce, DEVICE_LOCK_POLL_MS);
}

export function stopDeviceLockPolling() {
  if (_deviceLockPollTimer) clearInterval(_deviceLockPollTimer);
  _deviceLockPollTimer = null;
  _cachedDeviceLock = null;
}

// Fast, non-blocking read for UI purposes (opening the paywall). Falls
// back to a live check if polling hasn't produced a result yet (e.g.
// paywall opened in the first instant after sign-in, before the first poll
// tick resolves) so the UI is never left with no answer at all.
export async function getCachedDeviceLock() {
  if (_cachedDeviceLock !== null) return _cachedDeviceLock;
  return _pollDeviceLockOnce();
}

// Force an immediate refresh of the cache — call right after a sign-in /
// account switch so the paywall doesn't briefly show a stale result left
// over from the previous account on this device.
export function refreshDeviceLockNow() {
  return _pollDeviceLockOnce();
}

// Call ONLY after a purchase has been verified as genuinely successful
// (i.e. right where purchasePremium() already writes premium:true).
export async function claimDeviceForPurchase() {
  const u = auth.currentUser;
  if (!u) return;
  const deviceId = await getDeviceId();
  await setDoc(doc(db, 'deviceRegistry', deviceId), {
    purchasedByUid: u.uid,
    purchasedByEmail: u.email || null,
    claimedAt: new Date().toISOString(),
  }, { merge: true });
}

export const PREMIUM_LIMITS = {
  free:    { maxCourses: 5,  aiAssistant: false, explainWhy: false, reviewWrongAnswers: false, comboLesson: false },
  premium: { maxCourses: 10, aiAssistant: true,  explainWhy: true,  reviewWrongAnswers: true,  comboLesson: true  },
};

let _isPremium = false;
let _listeners = [];
let _rcConfigured = false;
let _unsubProfile = null;

export function isPremium() {
  return _isPremium;
}

export function limits() {
  return _isPremium ? PREMIUM_LIMITS.premium : PREMIUM_LIMITS.free;
}

// Fires immediately with the current value, then again on every change —
// so callers don't need a separate "read it once" path.
export function onPremiumChange(cb) {
  _listeners.push(cb);
  cb(_isPremium);
  return () => { _listeners = _listeners.filter((l) => l !== cb); };
}

function _setPremium(val) {
  if (val === _isPremium) return;
  _isPremium = val;
  _listeners.forEach((cb) => cb(_isPremium));
}

// Call once right after sign-in (mirrors initPushForCurrentUser in main.js).
export async function initPremiumForCurrentUser() {
  const u = auth.currentUser;
  if (!u) return;

  if (_unsubProfile) _unsubProfile();
  _unsubProfile = onSnapshot(doc(db, 'users', u.uid, 'learnProfile', 'main'), (snap) => {
    _setPremium(!!snap.data()?.premium);
  });

  // Start (or restart, for the new account) the background device-lock
  // poll so openPaywall() can read a fresh-ish cached value instantly
  // instead of blocking on a network read. refreshDeviceLockNow() inside
  // startDeviceLockPolling()'s first tick means a just-switched account
  // isn't left showing the previous account's cached result.
  startDeviceLockPolling();

  const Purchases = window.Capacitor?.Plugins?.Purchases;
  if (!Purchases) {
    console.warn('RevenueCat native plugin not found — did you run `npm install @revenuecat/purchases-capacitor && npx cap sync`?');
    return;
  }
  if (_rcConfigured) return;
  _rcConfigured = true;
  await Purchases.configure({ apiKey: RC_API_KEY_IOS, appUserID: u.uid });
}

export function resetPremiumState() {
  if (_unsubProfile) _unsubProfile();
  _unsubProfile = null;
  _rcConfigured = false;
  stopDeviceLockPolling();
  _setPremium(false);
}

// Returns true if the purchase went through, is genuinely tied to THIS
// account, and the entitlement is active. Throws on failure (including
// user cancelling — check err.userCancelled if the plugin sets it, and
// just don't show an error toast for that case) and on a cross-account
// entitlement mismatch (see the comment in restorePurchases below —
// StoreKit silently treats "buy" as "restore" for a non-consumable the
// Apple ID already owns, so purchasePackage() can succeed here too
// without this account actually having paid anything).
export async function purchasePremium() {
  const Purchases = window.Capacitor?.Plugins?.Purchases;
  if (!Purchases) throw new Error('Purchases not available on this platform.');

  const u = auth.currentUser;
  if (!u) throw new Error('Not signed in.');

  const { current } = await Purchases.getOfferings();
  const pkg = current?.availablePackages?.[0];
  if (!pkg) throw new Error('No premium package configured — check the RevenueCat Offering.');

  const { customerInfo } = await Purchases.purchasePackage({ aPackage: pkg });
  const active = !!customerInfo?.entitlements?.active?.[ENTITLEMENT_ID];
  if (!active) return false;

  const purchaserUid = customerInfo?.originalAppUserId;
  if (purchaserUid && purchaserUid !== u.uid) {
    throw new Error("This Apple ID already owns Premium on a different Kids Learning Lab account. Sign in to that account, or use Restore Purchases there.");
  }

  // Optimistic local write so the app reflects premium immediately —
  // the webhook confirms the same value into Firestore within seconds,
  // and is what every *other* device relies on.
  await setDoc(doc(db, 'users', u.uid, 'learnProfile', 'main'), { premium: true }, { merge: true });

  // Lock this physical device to this account now that a real purchase
  // has gone through — checkDeviceLock() in paywall.js reads this to
  // block a different account from getting Premium free on this device.
  await claimDeviceForPurchase();

  return active;
}

export async function restorePurchases() {
  const Purchases = window.Capacitor?.Plugins?.Purchases;
  if (!Purchases) throw new Error('Purchases not available on this platform.');

  const u = auth.currentUser;
  if (!u) throw new Error('Not signed in.');

  const { customerInfo } = await Purchases.restorePurchases();
  const active = !!customerInfo?.entitlements?.active?.[ENTITLEMENT_ID];
  if (!active) return false;

  // The entitlement being "active" only means SOME account on this Apple ID
  // bought Premium — Purchases.configure() re-identifies as whichever
  // Firebase account happens to be signed in on this device (see
  // initPremiumForCurrentUser above), so Apple/RevenueCat will report the
  // same active entitlement no matter which of a family's accounts is
  // currently open. Blindly trusting `active` here is exactly how a
  // second, never-purchased account could get Premium for free just by
  // tapping Restore while signed in on a device that's already paid.
  //
  // originalAppUserId is RevenueCat's record of which app-user-id (i.e.
  // Firebase uid, since that's what we pass as appUserID) was signed in
  // at the moment of the ORIGINAL purchase transaction, and it survives
  // being re-configured under a different uid later — so it's the one
  // field that actually answers "did THIS account buy this," rather than
  // "is this Apple ID entitled to it."
  const purchaserUid = customerInfo?.originalAppUserId;
  if (purchaserUid && purchaserUid !== u.uid) {
    throw new Error("This purchase belongs to a different Kids Learning Lab account on this Apple ID. Sign in to that account to restore it there.");
  }

  await setDoc(doc(db, 'users', u.uid, 'learnProfile', 'main'), { premium: true }, { merge: true });
  return active;
}

// ============================================================
// DAILY USAGE (games / lessons) — REMOVED
//
// Lesson/game gating no longer exists: everyone gets unlimited lessons and
// games, free or Premium. canUseToday()/bumpDailyUsage()/getDailyUsage()
// are kept as harmless stubs (always-allow / no-op / always-zero) purely so
// existing call sites in learn.js and main.js don't need to be ripped out
// one by one — they're inert now and safe to delete at leisure.
// ============================================================
export function getDailyUsage(_learnProfile) {
  return { date: '', lessons: 0, games: 0 };
}

export function canUseToday(_kind, _learnProfile) {
  return true;
}

export async function bumpDailyUsage(_kind) {
  return { date: '', lessons: 0, games: 0 };
}