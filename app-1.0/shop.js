// shop.js — Kids Learning Lab XP Shop data layer (no DOM)
//
// Pulled out as its own dependency-free module (same pattern as friends.js)
// because both main.js (Home stat button, Profile XP row, Avatar picker
// unlocked colors/emoji) and learn.js (Streak modal button, streak-pass
// consumption in applyStreakDecay, Personalized Review credit check,
// maxCourses/premium-day gating) need it, and main.js/learn.js cannot
// import each other directly.
//
// learnProfile (users/{uid}/learnProfile/main) gains these fields, all
// defaulted lazily by normalizeShopFields() so existing accounts without
// them don't break:
//   xp                    (already existed)
//   streakPassCount        number, 0..slotCap(), passes bought but not yet consumed
//   passDates               string[] "YYYY-MM-DD" — calendar days saved by a
//                            pass (rendered green instead of orange)
//   unlockedColors          string[] hex colors purchased beyond the 15 defaults
//   unlockedEmoji            string[] single emoji purchased beyond the 30 defaults
//   purchasedReviewCredits  number — bought Full Personalized Review credits not yet used
//   bankedPremiumDays        string[] "YYYY-MM-DD" — dates on which the user gets
//                            premium perks (except course creation) from a
//                            purchased Premium Day
//   extraCourseSlotBought   boolean — one-time-ever purchase
import { auth, db } from './firebase.js';
import { doc, getDoc, setDoc, increment, onSnapshot } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";
import { isPremium } from './premium.js';

// ============================================================
// SHARED LIVE STATE
// ============================================================
// A single onSnapshot listener on learnProfile/main, scoped to the shop
// fields only, so main.js (avatar picker's unlocked colors/emoji, XP shop
// entry points) and learn.js (streak-pass consumption, review credits) can
// each read a consistent, always-current copy without either of them
// owning the source of truth or duplicating the subscription. Call
// initShopStateForCurrentUser() once after sign-in (mirrors
// initPremiumForCurrentUser in premium.js); onShopStateChange() fires
// immediately with whatever's cached, then again on every server change.
let _shopState = normalizeShopFieldsShape();
let _shopListeners = [];
let _unsubShopState = null;

function normalizeShopFieldsShape() {
  return {
    xp: 0, streak: 0, streakPassCount: 0, passDates: [], unlockedColors: [],
    unlockedEmoji: [], purchasedReviewCredits: 0, bankedPremiumDays: [], extraCourseSlotBought: false,
  };
}

export function shopState() {
  return _shopState;
}

export function onShopStateChange(cb) {
  _shopListeners.push(cb);
  cb(_shopState);
  return () => { _shopListeners = _shopListeners.filter((l) => l !== cb); };
}

export function initShopStateForCurrentUser() {
  const u = auth.currentUser;
  if (!u) return;
  if (_unsubShopState) _unsubShopState();
  _unsubShopState = onSnapshot(doc(db, 'users', u.uid, 'learnProfile', 'main'), (snap) => {
    const data = snap.exists() ? snap.data() : {};
    _shopState = normalizeShopFields({ ...normalizeShopFieldsShape(), ...data });
    _shopListeners.forEach((cb) => cb(_shopState));
  }, (err) => {
    console.error('Shop state listener failed:', err);
  });
}

export function resetShopState() {
  if (_unsubShopState) _unsubShopState();
  _unsubShopState = null;
  _shopState = normalizeShopFieldsShape();
  _shopListeners.forEach((cb) => cb(_shopState));
}

export const SHOP_PRICES = {
  streakPass: 1500,
  avatarColor: 1000,
  avatarEmoji: 1000,
  streakProtection: 7500,
  fullPersonalizedReview: 2000, // "Full Personalized Review" credit (was "Personalized Review" — renamed, same price)
  premiumDay: 30000,
  extraCourseSlot: 30000,
};

// Max streak passes a user can hold at once (unused, banked passes) —
// separate from Streak Protection, which grants its 7 days immediately
// and doesn't touch this cap.
export function streakPassCap() {
  return isPremium() ? 4 : 2;
}

export function normalizeShopFields(learnProfile) {
  if (typeof learnProfile.streakPassCount !== 'number') learnProfile.streakPassCount = 0;
  if (!Array.isArray(learnProfile.passDates)) learnProfile.passDates = [];
  if (!Array.isArray(learnProfile.unlockedColors)) learnProfile.unlockedColors = [];
  if (!Array.isArray(learnProfile.unlockedEmoji)) learnProfile.unlockedEmoji = [];
  if (typeof learnProfile.purchasedReviewCredits !== 'number') learnProfile.purchasedReviewCredits = 0;
  if (!Array.isArray(learnProfile.bankedPremiumDays)) learnProfile.bankedPremiumDays = [];
  if (typeof learnProfile.extraCourseSlotBought !== 'boolean') learnProfile.extraCourseSlotBought = false;
  return learnProfile;
}

function todayStrLocal() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

function addDaysStr(baseStr, days) {
  const d = new Date(baseStr + 'T00:00:00');
  d.setDate(d.getDate() + days);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

// Re-reads the server's XP balance immediately before spending (rather
// than trusting the in-memory learnProfile) so two rapid purchases — e.g.
// two taps before the UI disables the button — can't both succeed against
// a stale cached balance.
async function spendXp(amount) {
  const u = auth.currentUser;
  if (!u) throw new Error('Not signed in.');
  const ref = doc(db, 'users', u.uid, 'learnProfile', 'main');
  const snap = await getDoc(ref);
  const currentXp = snap.exists() ? (snap.data().xp || 0) : 0;
  if (currentXp < amount) throw new Error('Not enough XP.');
  await setDoc(ref, { xp: increment(-amount) }, { merge: true });
  return currentXp - amount;
}

// Every purchase function below: spends XP first (throws + spends nothing
// if insufficient), applies its specific effect, returns the fields that
// changed so the caller can patch its in-memory learnProfile without a
// full re-read. All Firestore writes are merge:true so they can't clobber
// a concurrent XP award (badges.js, awardGameXp, etc).

export async function buyStreakPass(learnProfile) {
  const cap = streakPassCap();
  if (learnProfile.streakPassCount >= cap) throw new Error(`You already have the max of ${cap} streak passes.`);
  await spendXp(SHOP_PRICES.streakPass);
  const u = auth.currentUser;
  const newCount = learnProfile.streakPassCount + 1;
  await setDoc(doc(db, 'users', u.uid, 'learnProfile', 'main'), { streakPassCount: newCount }, { merge: true });
  return { streakPassCount: newCount };
}

export async function buyAvatarColor(learnProfile, hexColor) {
  await spendXp(SHOP_PRICES.avatarColor);
  const u = auth.currentUser;
  const unlockedColors = [...learnProfile.unlockedColors, hexColor];
  await setDoc(doc(db, 'users', u.uid, 'learnProfile', 'main'), { unlockedColors }, { merge: true });
  return { unlockedColors };
}

export async function buyAvatarEmoji(learnProfile, emoji) {
  await spendXp(SHOP_PRICES.avatarEmoji);
  const u = auth.currentUser;
  const unlockedEmoji = [...learnProfile.unlockedEmoji, emoji];
  await setDoc(doc(db, 'users', u.uid, 'learnProfile', 'main'), { unlockedEmoji }, { merge: true });
  return { unlockedEmoji };
}

// Applied immediately: covers the next 7 calendar days starting today, as
// pass-covered (green) days — independent of streakPassCount/its cap.
export async function buyStreakProtection(learnProfile) {
  await spendXp(SHOP_PRICES.streakProtection);
  const u = auth.currentUser;
  const today = todayStrLocal();
  const newDates = [];
  for (let i = 0; i < 7; i++) newDates.push(addDaysStr(today, i));
  const passDates = [...new Set([...learnProfile.passDates, ...newDates])];
  await setDoc(doc(db, 'users', u.uid, 'learnProfile', 'main'), { passDates }, { merge: true });
  return { passDates };
}

export async function buyFullPersonalizedReview(learnProfile) {
  await spendXp(SHOP_PRICES.fullPersonalizedReview);
  const u = auth.currentUser;
  const newCount = (learnProfile.purchasedReviewCredits || 0) + 1;
  await setDoc(doc(db, 'users', u.uid, 'learnProfile', 'main'), { purchasedReviewCredits: newCount }, { merge: true });
  return { purchasedReviewCredits: newCount };
}

// Redeemable starting the day AFTER purchase (per spec) — not today.
export async function buyPremiumDay(learnProfile) {
  await spendXp(SHOP_PRICES.premiumDay);
  const u = auth.currentUser;
  const startsDate = addDaysStr(todayStrLocal(), 1);
  const bankedPremiumDays = [...new Set([...learnProfile.bankedPremiumDays, startsDate])];
  await setDoc(doc(db, 'users', u.uid, 'learnProfile', 'main'), { bankedPremiumDays }, { merge: true });
  return { bankedPremiumDays };
}

export async function buyExtraCourseSlot(learnProfile) {
  if (learnProfile.extraCourseSlotBought) throw new Error('You already own this.');
  await spendXp(SHOP_PRICES.extraCourseSlot);
  const u = auth.currentUser;
  await setDoc(doc(db, 'users', u.uid, 'learnProfile', 'main'), { extraCourseSlotBought: true }, { merge: true });
  return { extraCourseSlotBought: true };
}

// True if `dateStr` (default today) has "banked premium day" perks active —
// everything premium EXCEPT course creation, which callers must gate
// separately (see isCourseCreationBlockedByBankedDay).
export function hasBankedPremiumPerks(learnProfile, dateStr) {
  const d = dateStr || todayStrLocal();
  return (learnProfile.bankedPremiumDays || []).includes(d);
}

export { todayStrLocal as shopTodayStr };