// shopUI.js — Kids Learning Lab XP Shop UI (DOM)
//
// Its own module (not folded into main.js or learn.js) for the same reason
// as paywall.js: both main.js (Home stat button, Profile XP row) and
// learn.js (Streak modal button) need to open the shop, and main.js/
// learn.js can't import each other. All Firestore writes/reads live in
// shop.js — this file only ever touches the DOM plus calls into shop.js.
import { auth } from './firebase.js';
import { isPremium } from './premium.js';
import { openPaywall } from './paywall.js';
import {
  SHOP_PRICES, streakPassCap, shopState, onShopStateChange,
  buyStreakPass, buyAvatarColor, buyAvatarEmoji, buyStreakProtection,
  buyFullPersonalizedReview, buyPremiumDay, buyExtraCourseSlot,
} from './shop.js';

const xpShopOverlay = document.getElementById('xpShopOverlay');
const xpShopExitBtn = document.getElementById('xpShopExitBtn');
const xpShopStreakCount = document.getElementById('xpShopStreakCount');
const xpShopXpBalance = document.getElementById('xpShopXpBalance');
const xpShopList = document.getElementById('xpShopList');

const shopColorPickerOverlay = document.getElementById('shopColorPickerOverlay');
const shopColorPickerBackBtn = document.getElementById('shopColorPickerBackBtn');
const shopColorPreview = document.getElementById('shopColorPreview');
const shopColorWheel = document.getElementById('shopColorWheel');
const shopColorCursor = document.getElementById('shopColorCursor');
const shopColorLightness = document.getElementById('shopColorLightness');
const shopColorHex = document.getElementById('shopColorHex');
const shopColorError = document.getElementById('shopColorError');
const shopColorBuyBtn = document.getElementById('shopColorBuyBtn');
const shopColorPriceLabel = document.getElementById('shopColorPriceLabel');

const shopEmojiPickerOverlay = document.getElementById('shopEmojiPickerOverlay');
const shopEmojiPickerBackBtn = document.getElementById('shopEmojiPickerBackBtn');
const shopEmojiGrid = document.getElementById('shopEmojiGrid');
const shopEmojiError = document.getElementById('shopEmojiError');
const shopEmojiBuyBtn = document.getElementById('shopEmojiBuyBtn');

if (shopColorPriceLabel) shopColorPriceLabel.textContent = String(SHOP_PRICES.avatarColor);

export function openXpShop() {
  if (!xpShopOverlay) {
    console.warn('XP Shop markup missing from index.html (#xpShopOverlay).');
    return;
  }
  document.dispatchEvent(new CustomEvent('kll:refreshHome'));
  renderShopList(shopStateFromWidgets());
  xpShopOverlay.classList.add('show');
}

// Reads the streak/XP numbers straight off other pages' own already-
// rendered DOM instead of a fresh Firestore read — same pattern
// loadHomeData() already uses to pull the latest episode title off the
// Listen tab's rendered list. XP comes from Settings/Profile's
// #profileXpCount specifically (not Home's #homeStatXpNum): that field is
// kept live via its own onSnapshot listener from the moment the app signs
// in, whereas Home's copy only exists once Home has actually rendered at
// least once — opening the XP Shop before ever visiting Home used to read
// #homeStatXpNum's unrendered "0" default and show 0 XP even though the
// real balance had already loaded elsewhere. Streak still falls back
// through Home/Learn's widgets since there's no equivalent always-live
// streak field on Profile. Everything else (streakPassCount,
// purchasedReviewCredits, etc, which have no on-page widget of their own)
// still comes from the live shop.js state.
function shopStateFromWidgets() {
  const base = shopState();
  const homeStreakEl = document.getElementById('homeStatStreakNum');
  const learnStreakEl = document.getElementById('learnStreakCount');
  const profileXpEl = document.getElementById('profileXpCount');

  const streakText = homeStreakEl?.textContent ?? learnStreakEl?.textContent;
  const xpText = profileXpEl?.textContent;

  const streak = streakText != null && streakText !== '' ? parseInt(streakText, 10) : NaN;
  const xp = xpText != null && xpText !== '' ? parseInt(xpText.replace(/,/g, ''), 10) : NaN;

  return {
    ...base,
    streak: Number.isFinite(streak) ? streak : base.streak,
    xp: Number.isFinite(xp) ? xp : base.xp,
  };
}
function closeXpShop() {
  xpShopOverlay?.classList.remove('show');
}
xpShopExitBtn?.addEventListener('click', closeXpShop);

// Keep the streak/XP banner and every item's afford-ability in sync live —
// so if XP changes while the shop happens to be open (finishing a lesson
// in another tab, a badge award, etc), buttons enable/disable immediately
// rather than only reflecting the balance from when the shop was opened.
// Streak/XP numbers themselves still come from the Home/Learn widgets
// (shopStateFromWidgets), not this Firestore-derived `state` directly.
onShopStateChange((state) => {
  if (xpShopOverlay?.classList.contains('show')) renderShopList(shopStateFromWidgets());
});

function fmtXp(n) {
  return n.toLocaleString();
}

function renderShopList(state) {
  if (!xpShopList) return;
  xpShopStreakCount.textContent = state.streak ?? 0;
  xpShopXpBalance.textContent = fmtXp(state.xp || 0);

  const cap = streakPassCap();
  const items = [
    {
      icon: 'bookmark',
      title: 'Streak Pass',
      sub: `One grace day — shows green on your streak calendar, only used if you miss a day. You have ${state.streakPassCount}/${cap}.`,
      price: SHOP_PRICES.streakPass,
      disabled: state.streakPassCount >= cap,
      disabledReason: `Max ${cap} at a time`,
      onBuy: async () => { await buyStreakPass(state); },
    },
    {
      icon: 'palette',
      title: 'New Avatar Color',
      sub: 'Unlock a custom color to use on your avatar.',
      price: SHOP_PRICES.avatarColor,
      onBuy: () => openShopColorPicker(),
      noSpend: true, // spending happens inside the color picker sub-page, not here
    },
    {
      icon: 'mood',
      title: 'New Avatar Emoji',
      sub: 'Unlock any emoji to use on your avatar.',
      price: SHOP_PRICES.avatarEmoji,
      onBuy: () => openShopEmojiPicker(),
      noSpend: true, // spending happens inside the emoji picker sub-page, not here
    },
    {
      icon: 'shield',
      title: 'Streak Protection',
      sub: 'Saves your streak for the next 7 days — like 7 streak passes, but used immediately. Best for vacations.',
      price: SHOP_PRICES.streakProtection,
      onBuy: async () => {
        await buyStreakProtection(state);
        return 'Applied! The next 7 days are covered on your streak calendar.';
      },
    },
    {
      icon: 'history_edu',
      title: '1 Full Personalized Review',
      sub: `You have ${state.purchasedReviewCredits || 0} credit${(state.purchasedReviewCredits || 0) === 1 ? '' : 's'}. Go to your Review Page and press Full Personalized Review — works once.`,
      price: SHOP_PRICES.fullPersonalizedReview,
      onBuy: async () => {
        await buyFullPersonalizedReview(state);
        return 'You can now go to your Review Page and press Full Personalized Review — it works once.';
      },
    },
    {
      icon: 'workspace_premium',
      title: '1 Day of Premium',
      sub: 'Unlimited lessons and games tomorrow (course creation not included).',
      price: SHOP_PRICES.premiumDay,
      onBuy: async () => {
        await buyPremiumDay(state);
        return 'Starting tomorrow, you\u2019ll have a full day of Premium!';
      },
    },
    {
      icon: 'add_box',
      title: '1 Extra Course Slot',
      sub: `One-time — lets you add ${isPremium() ? 11 : 6} courses instead of ${isPremium() ? 10 : 5}.`,
      price: SHOP_PRICES.extraCourseSlot,
      disabled: !!state.extraCourseSlotBought,
      owned: !!state.extraCourseSlotBought,
      onBuy: async () => { await buyExtraCourseSlot(state); },
    },
    {
      icon: 'star',
      title: 'Kids Learning Lab Premium',
      sub: 'Unlimited lessons, games, and more.',
      priceLabel: '$2.49',
      isRealMoney: true,
      onBuy: () => { closeXpShop(); openPaywall({}); },
    },
  ];

  xpShopList.innerHTML = '';
  items.forEach((item) => {
    const card = document.createElement('div');
    card.className = 'shop-item-card';

    const affordable = item.isRealMoney || (state.xp || 0) >= item.price;
    const canBuy = item.isRealMoney || (!item.disabled && affordable);

    const priceHtml = item.isRealMoney
      ? `<div class="shop-item-price">${item.priceLabel}</div>`
      : `<div class="shop-item-price${item.owned ? ' owned' : ''}">${item.owned ? 'Owned' : `${fmtXp(item.price)} XP`}</div>`;

    card.innerHTML = `
      <div class="shop-item-top">
        <div class="shop-item-icon"><span class="material-symbols-outlined">${item.icon}</span></div>
        <div>
          <div class="shop-item-title">${item.title}</div>
          <div class="shop-item-sub">${item.sub || ''}</div>
        </div>
        ${priceHtml}
      </div>
      ${item.owned ? '' : `<button type="button" class="shop-buy-btn${item.isRealMoney ? ' pink' : ''}" ${canBuy ? '' : 'disabled'}>${item.isRealMoney ? 'Go Premium' : (item.disabled ? (item.disabledReason || 'Unavailable') : (affordable ? 'Buy' : 'Not enough XP'))}</button>`}
      <div class="shop-item-error"></div>
    `;

    const buyBtn = card.querySelector('.shop-buy-btn');
    const errEl = card.querySelector('.shop-item-error');
    buyBtn?.addEventListener('click', async () => {
      buyBtn.disabled = true;
      const prevLabel = buyBtn.textContent;
      buyBtn.textContent = 'Processing…';
      errEl.textContent = '';
      try {
        const result = await item.onBuy();
        if (typeof result === 'string') errEl.textContent = result;
        // onShopStateChange will re-render the whole list with fresh data
        // once the Firestore write lands — no need to manually reset the
        // button here for the success path (item.noSpend cards handle
        // their own navigation instead of a spend).
      } catch (err) {
        errEl.textContent = err?.message || 'Something went wrong.';
        buyBtn.disabled = false;
        buyBtn.textContent = prevLabel;
      }
    });

    xpShopList.appendChild(card);
  });
}

// ============================================================
// CUSTOM COLOR PICKER (hue/saturation wheel + lightness slider)
// ============================================================
let shopColorHue = 220;
let shopColorSat = 80;
let shopColorLight = 50;

function hslToHex(h, s, l) {
  s /= 100; l /= 100;
  const k = (n) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  const toHex = (x) => Math.round(255 * x).toString(16).padStart(2, '0');
  return `#${toHex(f(0))}${toHex(f(8))}${toHex(f(4))}`.toUpperCase();
}

function updateShopColorPreview() {
  const hex = hslToHex(shopColorHue, shopColorSat, shopColorLight);
  shopColorPreview.style.background = hex;
  shopColorHex.textContent = hex;
  // Position the cursor on the wheel: inverse of pickColorFromWheelEvent's
  // mapping (radius = 110px for the 220px wheel).
  const radius = (shopColorSat / 100) * 110;
  const cssAngle = shopColorHue - 90; // undo the "+90 to align with gradient" step
  const atan2Angle = (cssAngle - 90) * (Math.PI / 180); // undo the "+90 for from-top" step
  const cx = 110 + radius * Math.cos(atan2Angle);
  const cy = 110 + radius * Math.sin(atan2Angle);
  shopColorCursor.style.left = `${cx}px`;
  shopColorCursor.style.top = `${cy}px`;
  return hex;
}

function pickColorFromWheelEvent(evt) {
  const rect = shopColorWheel.getBoundingClientRect();
  const point = evt.touches ? evt.touches[0] : evt;
  const cx0 = rect.left + rect.width / 2;
  const cy0 = rect.top + rect.height / 2;
  const x = point.clientX - cx0;
  const y = point.clientY - cy0;
  const radiusPx = rect.width / 2;
  const dist = Math.min(Math.sqrt(x * x + y * y), radiusPx);
  // conic-gradient(from 90deg, red, yellow, lime, cyan, blue, magenta, red)
  // means CSS angle 0deg (pointing up, i.e. atan2 angle -90deg) = red = hue 90.
  // Standard atan2 angle (0deg = pointing right, increasing clockwise) needs
  // +90 to become "CSS conic angle from top", then +90 again to land on the
  // gradient's starting hue offset, matching what's actually painted.
  let cssAngle = Math.atan2(y, x) * (180 / Math.PI) + 90;
  if (cssAngle < 0) cssAngle += 360;
  let hue = cssAngle + 90;
  if (hue >= 360) hue -= 360;
  shopColorHue = hue;
  shopColorSat = Math.min(100, (dist / radiusPx) * 100);
  updateShopColorPreview();
}

let _wheelDragging = false;
shopColorWheel?.addEventListener('pointerdown', (e) => {
  _wheelDragging = true;
  shopColorWheel.setPointerCapture?.(e.pointerId);
  pickColorFromWheelEvent(e);
});
shopColorWheel?.addEventListener('pointermove', (e) => { if (_wheelDragging) pickColorFromWheelEvent(e); });
shopColorWheel?.addEventListener('pointerup', (e) => {
  _wheelDragging = false;
  shopColorWheel.releasePointerCapture?.(e.pointerId);
});
shopColorWheel?.addEventListener('pointercancel', () => { _wheelDragging = false; });

shopColorLightness?.addEventListener('input', () => {
  shopColorLight = Number(shopColorLightness.value);
  updateShopColorPreview();
});

function openShopColorPicker() {
  shopColorError.textContent = '';
  shopColorHue = 220; shopColorSat = 80; shopColorLight = 50;
  shopColorLightness.value = '50';
  updateShopColorPreview();
  xpShopOverlay.classList.remove('show');
  shopColorPickerOverlay.classList.add('show');
}
shopColorPickerBackBtn?.addEventListener('click', () => {
  shopColorPickerOverlay.classList.remove('show');
  xpShopOverlay.classList.add('show');
});

shopColorBuyBtn?.addEventListener('click', async () => {
  shopColorError.textContent = '';
  shopColorBuyBtn.disabled = true;
  const prevLabel = shopColorBuyBtn.textContent;
  shopColorBuyBtn.textContent = 'Buying…';
  try {
    const hex = hslToHex(shopColorHue, shopColorSat, shopColorLight);
    await buyAvatarColor(shopState(), hex);
    shopColorPickerOverlay.classList.remove('show');
    xpShopOverlay.classList.add('show');
  } catch (err) {
    shopColorError.textContent = err?.message || 'Something went wrong.';
  } finally {
    shopColorBuyBtn.disabled = false;
    shopColorBuyBtn.textContent = prevLabel;
  }
});

// ============================================================
// EMOJI PICKER ("any emoji except inappropriate ones")
// ============================================================
// A broad, kid-safe emoji set — much wider than the 30 curated defaults in
// main.js's avatar picker, but still a curated allow-list rather than the
// full Unicode emoji range, so nothing inappropriate can slip in no matter
// what the learner searches for (there's no text search here at all).
const SHOP_EMOJI_CHOICES = [
  '😀', '😃', '😄', '😁', '😆', '😅', '🤣', '😂', '🙂', '😉', '😊', '😇', '🥰', '😍', '🤩', '😘',
  '😋', '😛', '😜', '🤪', '😎', '🥳', '🤗', '🤔', '🤨', '😐', '😴', '🥱', '😮', '😯', '😲', '🥺',
  '🦁', '🐯', '🐶', '🐱', '🐭', '🐹', '🐰', '🦊', '🐻', '🐼', '🐨', '🐵', '🐔', '🐧', '🐦', '🐤',
  '🦆', '🦅', '🦉', '🦇', '🐺', '🐗', '🐴', '🦄', '🐝', '🐛', '🦋', '🐌', '🐞', '🐢', '🐍', '🦖',
  '🦕', '🐙', '🦑', '🦐', '🦀', '🐡', '🐠', '🐟', '🐬', '🐳', '🐋', '🦈', '🐊', '🐆', '🦓', '🦍',
  '🐘', '🦛', '🦏', '🐪', '🐫', '🦒', '🦘', '🐃', '🐂', '🐄', '🐎', '🐖', '🐐', '🐑', '🦙', '🐓',
  '🦃', '🦚', '🦜', '🦢', '🦩', '🐇', '🦔', '🦇', '🐿️', '🌵', '🎄', '🌲', '🌳', '🌴', '🌱', '🌿',
  '☘️', '🍀', '🎍', '🎋', '🍃', '🍂', '🍁', '🍄', '🌾', '💐', '🌷', '🌹', '🥀', '🌺', '🌸', '🌼',
  '🌻', '🌞', '🌝', '🌛', '🌟', '⭐', '✨', '⚡', '🔥', '🌈', '☀️', '⛅', '☁️', '❄️', '⛄', '🌊',
  '🚀', '🛸', '🚁', '✈️', '🚂', '🚗', '🚕', '🚙', '🚌', '🏎️', '🚓', '🚑', '🚒', '🚲', '🛴', '⛵',
  '⚽', '🏀', '🏈', '⚾', '🎾', '🏐', '🏉', '🎱', '🏓', '🏸', '🥊', '🥋', '🎿', '🛹', '🛼', '🎯',
  '🎮', '🕹️', '🎲', '🧩', '🎨', '🎭', '🎪', '🎬', '🎤', '🎧', '🎸', '🎹', '🥁', '🎺', '🎷', '🎻',
  '📚', '📖', '✏️', '🖍️', '🎒', '🔭', '🔬', '🧪', '🧲', '💡', '🔦', '🕯️', '🧭', '🗺️', '🏆', '🥇',
  '🎁', '🎈', '🎉', '🎊', '🎀', '🍎', '🍊', '🍋', '🍌', '🍉', '🍇', '🍓', '🫐', '🍈', '🍒', '🍑',
  '🥭', '🍍', '🥥', '🥝', '🍅', '🥕', '🌽', '🍕', '🍔', '🌭', '🥪', '🌮', '🌯', '🍦', '🍩', '🍪',
  '🎂', '🍰', '🧁', '🍭', '🍬', '🍫', '🍿', '🥤', '🧃', '🌙', '💧', '🫧', '🎯', '🧸', '🪁', '🪀',
];

let shopSelectedEmoji = null;

function buildShopEmojiGrid() {
  if (shopEmojiGrid.childElementCount) return; // built once
  SHOP_EMOJI_CHOICES.forEach((emoji) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'shop-emoji-picker-btn';
    btn.textContent = emoji;
    btn.dataset.emoji = emoji;
    btn.addEventListener('click', () => {
      shopSelectedEmoji = emoji;
      shopEmojiGrid.querySelectorAll('.shop-emoji-picker-btn').forEach((b) => {
        b.classList.toggle('selected', b.dataset.emoji === emoji);
      });
      shopEmojiBuyBtn.disabled = false;
      shopEmojiError.textContent = '';
    });
    shopEmojiGrid.appendChild(btn);
  });
}

function openShopEmojiPicker() {
  buildShopEmojiGrid();
  shopSelectedEmoji = null;
  shopEmojiError.textContent = '';
  shopEmojiBuyBtn.disabled = true;
  shopEmojiGrid.querySelectorAll('.shop-emoji-picker-btn.selected').forEach((b) => b.classList.remove('selected'));
  xpShopOverlay.classList.remove('show');
  shopEmojiPickerOverlay.classList.add('show');
}
shopEmojiPickerBackBtn?.addEventListener('click', () => {
  shopEmojiPickerOverlay.classList.remove('show');
  xpShopOverlay.classList.add('show');
});

shopEmojiBuyBtn?.addEventListener('click', async () => {
  if (!shopSelectedEmoji) return;
  shopEmojiError.textContent = '';
  shopEmojiBuyBtn.disabled = true;
  const prevLabel = shopEmojiBuyBtn.textContent;
  shopEmojiBuyBtn.textContent = 'Buying…';
  try {
    await buyAvatarEmoji(shopState(), shopSelectedEmoji);
    shopEmojiPickerOverlay.classList.remove('show');
    xpShopOverlay.classList.add('show');
  } catch (err) {
    shopEmojiError.textContent = err?.message || 'Something went wrong.';
    shopEmojiBuyBtn.disabled = false;
    shopEmojiBuyBtn.textContent = prevLabel;
  }
});