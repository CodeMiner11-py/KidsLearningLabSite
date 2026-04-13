// app.js — Shared utilities for Kids Learning Lab PWA

window.KLL = window.KLL || {};

// ─── Favorites ────────────────────────────────────────────────────────────────
const FAV_KEY = 'kll-favorites';

function getFavorites() {
  try { return JSON.parse(localStorage.getItem(FAV_KEY)) || []; }
  catch { return []; }
}

function saveFavorites(favs) {
  localStorage.setItem(FAV_KEY, JSON.stringify(favs));
}

function isFavorite(id) {
  return getFavorites().some(f => f.id === id);
}

function toggleFavorite(episode) {
  let favs = getFavorites();
  if (favs.some(f => f.id === episode.id)) {
    favs = favs.filter(f => f.id !== episode.id);
  } else {
    favs.unshift({ id: episode.id, title: episode.title, art: episode.art,
                   duration: episode.duration, pubDateFormatted: episode.pubDateFormatted });
  }
  saveFavorites(favs);
  return !favs.some(f => f.id === episode.id) === false; // return new state
}

// ─── Service Worker ────────────────────────────────────────────────────────────
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch(() => {});
  });
}

// ─── Share ─────────────────────────────────────────────────────────────────────
async function shareEpisode(title, url) {
  if (navigator.share) {
    try {
      await navigator.share({ title: `${title} — Kids Learning Lab`, url });
      return;
    } catch {}
  }
  try {
    await navigator.clipboard.writeText(url);
    showToast('Link copied to clipboard');
  } catch {
    showToast('Copy this link: ' + url);
  }
}

// ─── Toast ─────────────────────────────────────────────────────────────────────
function showToast(msg) {
  let toast = document.getElementById('kll-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'kll-toast';
    toast.style.cssText = `
      position:fixed;bottom:90px;left:50%;transform:translateX(-50%) translateY(20px);
      background:#1565C0;color:#fff;padding:10px 20px;border-radius:24px;
      font-size:13px;font-family:inherit;opacity:0;transition:all .25s;
      z-index:9999;white-space:nowrap;box-shadow:0 4px 16px rgba(0,0,0,.2);
    `;
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.style.opacity = '1';
  toast.style.transform = 'translateX(-50%) translateY(0)';
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(-50%) translateY(20px)';
  }, 2500);
}

// ─── Active nav highlight ───────────────────────────────────────────────────────
function setActiveNav() {
  const page = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-item').forEach(el => {
    el.classList.toggle('active', el.dataset.page === page);
  });
}

// ─── Episode art fallback ───────────────────────────────────────────────────────
function onArtError(img) {
  img.style.display = 'none';
  const parent = img.parentElement;
  if (parent && !parent.querySelector('.art-fallback')) {
    const fb = document.createElement('span');
    fb.className = 'material-icons-round art-fallback';
    fb.textContent = 'headphones';
    parent.appendChild(fb);
  }
}

window.KLL = { ...window.KLL, getFavorites, saveFavorites, isFavorite, toggleFavorite, shareEpisode, showToast, setActiveNav, onArtError };
