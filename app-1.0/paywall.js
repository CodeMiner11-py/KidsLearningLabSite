// paywall.js — Kids Learning Lab full-page Premium paywall
//
// Deliberately its own module (not folded into main.js or learn.js) so both
// can open it without creating a circular import between them. All it does
// is own the paywall DOM + RevenueCat purchase button; premium.js remains
// the single source of truth for isPremium()/limits().
import { PREMIUM_LIMITS, purchasePremium, restorePurchases, isPremium, onPremiumChange, checkDeviceLock, getCachedDeviceLock } from './premium.js';
import { maybeShowOverlay } from './firstTimeOverlays.js';

const paywallOverlay = document.getElementById('paywallOverlay');
const paywallBenefits = document.getElementById('paywallBenefits');
const paywallReasonBanner = document.getElementById('paywallReasonBanner');
const paywallReasonText = document.getElementById('paywallReasonText');
const paywallTitle = document.getElementById('paywallTitle');
const paywallSub = document.getElementById('paywallSub');
const paywallPriceBtn = document.getElementById('paywallPriceBtn');
const paywallPriceBtnLabel = document.getElementById('paywallPriceBtnLabel');
const paywallError = document.getElementById('paywallError');
const paywallExitBtn = document.getElementById('paywallExitBtn');
const paywallMaybeLaterBtn = document.getElementById('paywallMaybeLaterBtn');
const paywallRestoreBtn = document.getElementById('paywallRestoreBtn');

const paywallConfirmModalOverlay = document.getElementById('paywallConfirmModalOverlay');
const paywallConfirmModalConfirmBtn = document.getElementById('paywallConfirmModalConfirmBtn');
const paywallConfirmModalCancelBtn = document.getElementById('paywallConfirmModalCancelBtn');
const paywallRestartingModalOverlay = document.getElementById('paywallRestartingModalOverlay');
const paywallRestartingFill = document.getElementById('paywallRestartingFill');

const deviceLockedModalOverlay = document.getElementById('deviceLockedModalOverlay');
const deviceLockedModalBody = document.getElementById('deviceLockedModalBody');
const deviceLockedModalOkBtn = document.getElementById('deviceLockedModalOkBtn');

const connectingModalOverlay = document.getElementById('connectingModalOverlay');

const DEFAULT_TITLE = 'Kids Learning Lab Premium';
const DEFAULT_SUB = 'A one-time purchase unlocks unlimited learning for you or your family.';
const DEFAULT_PRICE_LABEL = 'Upgrade to Premium';

let _onUpgradeCallback = null; // optional, fired once the purchase actually succeeds
let _offeringPkg = null;       // cached RevenueCat package, so Restore/purchase don't refetch every open

// Benefit rows: sourced from PREMIUM_LIMITS so this can never drift out of
// sync with the actual free/premium caps enforced elsewhere in the app.
const f = PREMIUM_LIMITS.free;
const BENEFIT_ROWS = [
  { icon: 'library_books', label: 'Courses at once', free: `${f.maxCourses}`, premium: `${PREMIUM_LIMITS.premium.maxCourses}` },
  { icon: 'local_fire_department', label: 'Streak Pass slots', free: '2', premium: '4' },
  { icon: 'auto_awesome', label: 'AI Assistant', free: false, premium: true },
  { icon: 'psychology', label: 'Explain My Answer', free: false, premium: true },
  { icon: 'history_edu', label: 'Personalized Review', free: false, premium: true },
  { icon: 'auto_fix_high', label: 'Combo Lessons', free: false, premium: true },
];

function renderBenefits() {
  if (!paywallBenefits) return;
  paywallBenefits.innerHTML = BENEFIT_ROWS.map((row) => `
    <div class="paywall-benefit-row">
      <div class="paywall-benefit-icon"><span class="material-symbols-outlined">${row.icon}</span></div>
      <div class="paywall-benefit-label">${row.label}</div>
      <div class="paywall-benefit-free">${typeof row.free === 'boolean' ? (row.free ? '✓' : '—') : row.free}</div>
      <div class="paywall-benefit-premium">${typeof row.premium === 'boolean' ? (row.premium ? '✓' : '—') : row.premium}</div>
    </div>
  `).join('');
}
renderBenefits();

function closePaywall() {
  if (!paywallOverlay) return;
  paywallOverlay.classList.remove('show');
  paywallError.textContent = '';
  _onUpgradeCallback = null;
}

// opts:
//   reason       — short banner text explaining *why* the paywall showed up
//                  (e.g. "You reached the course limit. Upgrade to continue").
//                  Omit for the generic "just browsing premium" case.
//   title, sub   — override the default headline/subhead
//   onUpgrade    — called once, right after a successful purchase/restore
async function openPaywall(opts = {}) {
  if (!paywallOverlay) {
    console.warn('Paywall markup missing from index.html (#paywallOverlay).');
    return;
  }
  if (isPremium()) return; // nothing to sell

  // Before showing anything purchase-related, make sure this device
  // hasn't already been used to buy Premium on a different account.
  //
  // This reads the 30s-background-refreshed cache instead of doing a live
  // network read, so opening the paywall is instant instead of blocking on
  // Firestore. That's fine here because opening the paywall UI doesn't
  // spend any money — the value only needs to be roughly current. The
  // live, uncached checkDeviceLock() is still called again right before
  // the purchase actually fires, in runPurchase() below; THAT check is the
  // one that actually has to be fresh, not this one.
  const lockedEmail = await getCachedDeviceLock();
  if (lockedEmail) {
    openDeviceLockedModal(lockedEmail);
    return;
  }

  const { reason, title, sub, onUpgrade } = opts;

  if (reason) {
    paywallReasonText.textContent = reason;
    paywallReasonBanner.style.display = 'flex';
  } else {
    paywallReasonBanner.style.display = 'none';
  }

  paywallTitle.textContent = title || DEFAULT_TITLE;
  paywallSub.textContent = sub || DEFAULT_SUB;
  paywallError.textContent = '';
  _onUpgradeCallback = typeof onUpgrade === 'function' ? onUpgrade : null;

  paywallOverlay.classList.add('show');
  loadPrice();
}

function showConnectingModal() {
  connectingModalOverlay?.classList.add('show');
}

function hideConnectingModal() {
  connectingModalOverlay?.classList.remove('show');
}

function openDeviceLockedModal(maskedEmail) {
  if (!deviceLockedModalOverlay) {
    console.warn('Device-locked modal markup missing from index.html (#deviceLockedModalOverlay).');
    return;
  }
  deviceLockedModalBody.textContent = `You've already purchased Kids Learning Lab Premium on this device, on the account ${maskedEmail}.`;
  deviceLockedModalOverlay.classList.add('show');
}

function closeDeviceLockedModal() {
  deviceLockedModalOverlay?.classList.remove('show');
}

deviceLockedModalOkBtn?.addEventListener('click', closeDeviceLockedModal);

async function loadPrice() {
  paywallPriceBtnLabel.textContent = DEFAULT_PRICE_LABEL;
  const Purchases = window.Capacitor?.Plugins?.Purchases;
  if (!Purchases) return; // web/dev preview — button still works, just shows the generic label

  try {
    const { current } = await Purchases.getOfferings();
    _offeringPkg = current?.availablePackages?.[0] || null;
    const priceString = _offeringPkg?.product?.priceString;
    if (priceString) {
      const period = _offeringPkg?.product?.subscriptionPeriod ? '/mo' : '';
      paywallPriceBtnLabel.textContent = `Upgrade — ${priceString}${period}`;
    }
  } catch (err) {
    console.warn('Could not load RevenueCat offering price:', err);
  }
}

paywallExitBtn?.addEventListener('click', closePaywall);
paywallMaybeLaterBtn?.addEventListener('click', closePaywall);

// Tapping the price button doesn't purchase right away — it opens a
// confirm step first, and only THAT step's Continue button actually calls
// purchasePremium() (which is what triggers the native iOS purchase
// sheet). This gives the learner (or the parent watching over their
// shoulder) one extra explicit "yes, buy this" tap before the OS dialog
// with the real charge shows up.
paywallPriceBtn?.addEventListener('click', () => {
  if (!paywallConfirmModalOverlay) {
    // No confirm-modal markup in the DOM for some reason — fail open
    // rather than leaving purchasing completely broken.
    runPurchase();
    return;
  }
  paywallConfirmModalOverlay.classList.add('show');
});

paywallConfirmModalCancelBtn?.addEventListener('click', () => {
  paywallConfirmModalOverlay.classList.remove('show');
});

paywallConfirmModalConfirmBtn?.addEventListener('click', async () => {
  paywallConfirmModalConfirmBtn.disabled = true;
  const prevLabel = paywallConfirmModalConfirmBtn.textContent;
  paywallConfirmModalConfirmBtn.textContent = 'Opening…';
  try {
    await runPurchase();
  } finally {
    paywallConfirmModalOverlay.classList.remove('show');
    paywallConfirmModalConfirmBtn.disabled = false;
    paywallConfirmModalConfirmBtn.textContent = prevLabel;
  }
});

// Shared by the confirm-modal Continue button and the no-confirm-markup
// fallback above — actually calls into RevenueCat (native iOS purchase
// sheet), then on a real, confirmed purchase, reloads the whole app so
// every module picks up the new premium state fresh rather than trying to
// patch a dozen already-rendered screens live.
async function runPurchase() {
  paywallError.textContent = '';

  // Re-check right before purchasing — covers the case where the paywall
  // was already open and the signed-in account changed underneath it. This
  // one stays a LIVE, uncached read (unlike the cached one openPaywall()
  // uses) since it's the actual gate on an actual charge.
  showConnectingModal();
  const lockedEmail = await checkDeviceLock();
  hideConnectingModal();
  if (lockedEmail) {
    closePaywall();
    openDeviceLockedModal(lockedEmail);
    return;
  }

  paywallPriceBtn.disabled = true;
  const prevLabel = paywallPriceBtnLabel.textContent;
  paywallPriceBtnLabel.textContent = 'Processing…';

  try {
    const active = await purchasePremium();
    if (active) {
      await reloadAppForPremium();
    } else {
      paywallError.textContent = "Purchase didn't complete. Try again.";
    }
  } catch (err) {
    if (!err?.userCancelled) {
      paywallError.textContent = err?.message || 'Something went wrong with the purchase.';
    }
  } finally {
    paywallPriceBtn.disabled = false;
    paywallPriceBtnLabel.textContent = prevLabel;
  }
}

// A hard reload is deliberate here, not a shortcut: premium state gates
// dozens of already-mounted things (daily limits, the AI Assistant FAB,
// course-slot math, the paywall's own benefit table, RevenueCat's cached
// entitlement) that would otherwise need to be individually re-checked
// live. Reloading guarantees every one of them boots up already knowing
// the account is premium, the same as if they'd just signed back in.
//
// The welcome overlay has to be shown and given time to actually be seen
// BEFORE the reload fires — reloading first would tear down the DOM (and
// the overlay with it) before the learner ever sees it, same bug that used
// to only affect the fresh-purchase path (Restore already got this right).
async function reloadAppForPremium() {
  closePaywall();
  await maybeShowOverlay('premiumWelcome');

  if (!paywallRestartingModalOverlay) {
    window.location.reload();
    return;
  }
  paywallRestartingModalOverlay.classList.add('show');
  if (paywallRestartingFill) {
    paywallRestartingFill.style.transition = 'none';
    paywallRestartingFill.style.width = '0%';
    requestAnimationFrame(() => {
      paywallRestartingFill.style.transition = 'width 1.4s linear';
      paywallRestartingFill.style.width = '100%';
    });
  }
  setTimeout(() => window.location.reload(), 1500);
}

paywallRestoreBtn?.addEventListener('click', async () => {
  paywallError.textContent = '';
  paywallRestoreBtn.disabled = true;
  const prevLabel = paywallRestoreBtn.textContent;
  paywallRestoreBtn.textContent = 'Restoring…';

  try {
    const active = await restorePurchases();
    if (active) {
      const cb = _onUpgradeCallback;
      closePaywall();
      maybeShowOverlay('premiumWelcome');
      cb?.();
    } else {
      paywallError.textContent = 'No active purchase found for this account.';
    }
  } catch (err) {
    paywallError.textContent = err?.message || 'Could not restore purchases.';
  } finally {
    paywallRestoreBtn.disabled = false;
    paywallRestoreBtn.textContent = prevLabel;
  }
});

// If premium flips on while the paywall happens to be open (e.g. webhook
// confirms on another device mid-session), close it out from under them.
onPremiumChange((premium) => {
  if (premium && paywallOverlay?.classList.contains('show')) closePaywall();
});

export { openPaywall, closePaywall };