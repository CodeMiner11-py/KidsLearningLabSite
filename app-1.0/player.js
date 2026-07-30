// player.js — Kids Learning Lab podcast player

const FEED_WORKER_URL = 'https://getrssfeed.nameless-cherry-998c.workers.dev/';
const CACHE_KEY = 'kll_podcast_feed_cache_v1';
const CACHE_TTL_MS = 30 * 60 * 1000; // 30 minutes

// ---- DOM refs ----
const listenStatus = document.getElementById('listenStatus');
const listenEpisodeList = document.getElementById('listenEpisodeList');

const miniPlayer = document.getElementById('miniPlayer');
const miniPlayerArt = document.getElementById('miniPlayerArt');
const miniPlayerTitle = document.getElementById('miniPlayerTitle');
const miniPlayerPlayPause = document.getElementById('miniPlayerPlayPause');
const miniPlayerPlayPauseIcon = document.getElementById('miniPlayerPlayPauseIcon');
const miniPlayerClose = document.getElementById('miniPlayerClose');

const episodeModalOverlay = document.getElementById('episodeModalOverlay');
const episodeModalCloseBtn = document.getElementById('episodeModalCloseBtn');
const episodeModalArt = document.getElementById('episodeModalArt');
const episodeModalTitle = document.getElementById('episodeModalTitle');
const episodeModalMeta = document.getElementById('episodeModalMeta');
const episodeModalPlayBtn = document.getElementById('episodeModalPlayBtn');
const episodeModalPlayIcon = document.getElementById('episodeModalPlayIcon');
const episodeModalDescription = document.getElementById('episodeModalDescription');
const playerCurrentTime = document.getElementById('playerCurrentTime');
const playerDuration = document.getElementById('playerDuration');
const playerSeekBar = document.getElementById('playerSeekBar');
const playerRewindBtn = document.getElementById('playerRewindBtn');
const playerForwardBtn = document.getElementById('playerForwardBtn');
const playerPrevEpisodeBtn = document.getElementById('playerPrevEpisodeBtn');
const playerNextEpisodeBtn = document.getElementById('playerNextEpisodeBtn');

const listenNavBtn = document.querySelector('.nav-btn[data-page="listen"]');
const authScreenEl = document.getElementById('auth-screen');

// ---- State ----
let episodes = [];          // newest -> oldest, as returned by the feed
let hasRenderedOnce = false;
let fetchInFlight = false;
let currentIndex = -1;      // index into episodes[] currently loaded in the audio element
let detailIndex = -1;       // index currently shown in the episode detail modal
let isSeeking = false;      // true while the user is dragging the seek bar

const audio = new Audio();
audio.preload = 'metadata';

// ============================================================
// LOADING THE FEED
// ============================================================
async function loadEpisodes() {
  if (fetchInFlight) return;

  // Show cached data instantly if we have nothing on screen yet
  if (!hasRenderedOnce) {
    const cached = readCache();
    if (cached && cached.episodes && cached.episodes.length) {
      episodes = cached.episodes;
      renderEpisodeList();
      hasRenderedOnce = true;
    } else {
      listenStatus.style.display = 'block';
      listenStatus.textContent = 'Loading episodes…';
    }
  }

  fetchInFlight = true;
  try {
    const res = await fetch(FEED_WORKER_URL);
    if (!res.ok) throw new Error('Feed request failed');
    const data = await res.json();
    const fresh = data.episodes || [];

    if (!hasRenderedOnce || !isSameFeed(fresh, episodes)) {
      applyFreshEpisodes(fresh);
      hasRenderedOnce = true;
    }
    writeCache(data);
  } catch (err) {
    if (!hasRenderedOnce) {
      listenStatus.textContent = "Couldn't load episodes. Check your connection and try again.";
    }
    console.error('Podcast feed load failed:', err);
  } finally {
    fetchInFlight = false;
  }
}

function isSameFeed(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
  return JSON.stringify(a) === JSON.stringify(b);
}

// Swap in a freshly-fetched episode list, re-locating whatever's currently
// playing/open by its episode id (not array position) so a reordered or
// updated feed can't silently start pointing at the wrong episode.
function applyFreshEpisodes(fresh) {
  const playingId = currentIndex >= 0 ? episodes[currentIndex]?.id : null;
  const detailId = detailIndex >= 0 ? episodes[detailIndex]?.id : null;

  episodes = fresh;

  currentIndex = playingId ? episodes.findIndex((e) => e.id === playingId) : -1;
  detailIndex = detailId ? episodes.findIndex((e) => e.id === detailId) : -1;

  renderEpisodeList();
  updateMiniPlayer();
  if (detailIndex >= 0 && episodeModalOverlay.classList.contains('show')) {
    updateEpisodeModalControls();
  }
}

function readCache() {
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (Date.now() - parsed.timestamp > CACHE_TTL_MS) return null;
    return parsed.data;
  } catch {
    return null;
  }
}

function writeCache(data) {
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify({ timestamp: Date.now(), data }));
  } catch {
    // ignore quota errors
  }
}

// ============================================================
// RENDERING THE EPISODE LIST
// ============================================================
function renderEpisodeList() {
  if (!episodes.length) {
    listenStatus.textContent = 'No episodes found.';
    return;
  }
  listenStatus.style.display = 'none';

  listenEpisodeList.innerHTML = episodes.map((ep, i) => `
    <div class="episode-card${i === currentIndex ? ' playing' : ''}" data-index="${i}">
      <img src="${escapeAttr(ep.image)}" alt="">
      <div class="episode-card-info">
        <div class="episode-card-title">${escapeHtml(ep.title)}</div>
        <div class="episode-card-meta">${formatDate(ep.pubDate)} · ${ep.duration || ''}</div>
      </div>
      <div class="episode-card-play">
        <span class="material-symbols-outlined">${i === currentIndex && !audio.paused ? 'pause' : 'play_arrow'}</span>
      </div>
    </div>
  `).join('');

  listenEpisodeList.querySelectorAll('.episode-card').forEach((card) => {
    card.addEventListener('click', () => {
      openEpisodeDetail(Number(card.dataset.index));
    });
  });
}

function formatDate(pubDate) {
  const d = new Date(pubDate);
  if (isNaN(d)) return '';
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

function escapeHtml(str) {
  return (str || '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function escapeAttr(str) {
  return escapeHtml(str);
}

// ============================================================
// EPISODE DETAIL MODAL
// ============================================================
function openEpisodeDetail(index) {
  const ep = episodes[index];
  if (!ep) return;
  detailIndex = index;

  episodeModalArt.src = ep.image;
  episodeModalTitle.textContent = ep.title;
  episodeModalMeta.textContent = `${formatDate(ep.pubDate)} · ${ep.duration || ''}`;
  episodeModalDescription.innerHTML = ep.description || '';

  updateEpisodeModalControls();
  episodeModalOverlay.classList.add('show');
}

function closeEpisodeDetail() {
  episodeModalOverlay.classList.remove('show');
}

function updateEpisodeModalControls() {
  const ep = episodes[detailIndex];
  if (!ep) return;

  const isLoadedHere = detailIndex === currentIndex;
  const isCurrentAndPlaying = isLoadedHere && !audio.paused;
  episodeModalPlayIcon.textContent = isCurrentAndPlaying ? 'pause' : 'play_arrow';

  const durationSeconds = (isLoadedHere && isFinite(audio.duration))
    ? audio.duration
    : parseDurationToSeconds(ep.duration);
  const currentSeconds = isLoadedHere ? audio.currentTime : 0;

  playerDuration.textContent = formatTime(durationSeconds);
  playerSeekBar.max = durationSeconds || 0;
  if (!isSeeking) {
    playerSeekBar.value = currentSeconds;
    playerCurrentTime.textContent = formatTime(currentSeconds);
  }

  playerPrevEpisodeBtn.disabled = detailIndex <= 0;
  playerNextEpisodeBtn.disabled = detailIndex >= episodes.length - 1;
}

episodeModalCloseBtn.addEventListener('click', closeEpisodeDetail);
episodeModalOverlay.addEventListener('click', (e) => {
  if (e.target === episodeModalOverlay) closeEpisodeDetail();
});

episodeModalPlayBtn.addEventListener('click', () => {
  if (detailIndex === currentIndex) {
    audio.paused ? resumePlayback() : pausePlayback();
  } else {
    playEpisode(detailIndex);
  }
});

// ---- Rewind / forward 10s ----
playerRewindBtn.addEventListener('click', () => seekBy(-10));
playerForwardBtn.addEventListener('click', () => seekBy(10));

function seekBy(deltaSeconds) {
  ensureLoadedThenSeek((audio, base) => clamp(base + deltaSeconds, 0, audio.duration || Infinity));
}

// ---- Drag-to-seek ----
playerSeekBar.addEventListener('input', () => {
  isSeeking = true;
  playerCurrentTime.textContent = formatTime(Number(playerSeekBar.value));
});
playerSeekBar.addEventListener('change', () => {
  const target = Number(playerSeekBar.value);
  ensureLoadedThenSeek(() => target);
  isSeeking = false;
});

// If the episode being viewed isn't the one loaded in the audio element yet,
// start it first, then apply the seek once real duration/timing is known.
function ensureLoadedThenSeek(computeTarget) {
  if (detailIndex !== currentIndex) {
    playEpisode(detailIndex);
    audio.addEventListener('loadedmetadata', function once() {
      audio.currentTime = clamp(computeTarget(audio, 0), 0, audio.duration || Infinity);
      audio.removeEventListener('loadedmetadata', once);
    }, { once: true });
  } else {
    audio.currentTime = clamp(computeTarget(audio, audio.currentTime), 0, audio.duration || Infinity);
  }
}

// ---- Previous / next episode ----
playerPrevEpisodeBtn.addEventListener('click', () => skipToAdjacentEpisode(-1));
playerNextEpisodeBtn.addEventListener('click', () => skipToAdjacentEpisode(1));

function skipToAdjacentEpisode(direction) {
  const baseIndex = detailIndex >= 0 ? detailIndex : currentIndex;
  if (baseIndex < 0) return;
  const targetIndex = baseIndex + direction;
  if (targetIndex < 0 || targetIndex >= episodes.length) return;
  playEpisode(targetIndex);
  openEpisodeDetail(targetIndex);
}

function clamp(v, min, max) {
  return Math.min(Math.max(v, min), max);
}

function parseDurationToSeconds(str) {
  if (!str) return 0;
  const parts = str.split(':').map(Number);
  if (parts.some(isNaN)) return 0;
  return parts.reduce((acc, val) => acc * 60 + val, 0);
}

function formatTime(seconds) {
  if (!isFinite(seconds) || seconds < 0) return '0:00';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const mm = h > 0 ? String(m).padStart(2, '0') : String(m);
  const ss = String(s).padStart(2, '0');
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

// ============================================================
// PLAYBACK
// ============================================================
function playEpisode(index) {
  const ep = episodes[index];
  if (!ep) return;
  currentIndex = index;
  audio.src = ep.audioUrl;
  audio.play().catch((err) => console.warn('Playback failed:', err));
  refreshAllPlayUI();
  updateMediaSession(ep);
}

function pausePlayback() {
  audio.pause();
  refreshAllPlayUI();
}

function resumePlayback() {
  if (currentIndex < 0) return;
  audio.play().catch((err) => console.warn('Playback failed:', err));
  refreshAllPlayUI();
}

function stopPlayback() {
  audio.pause();
  audio.currentTime = 0;
  audio.removeAttribute('src');
  currentIndex = -1;
  refreshAllPlayUI();
  if ('mediaSession' in navigator) {
    navigator.mediaSession.playbackState = 'none';
    navigator.mediaSession.metadata = null;
  }
}

audio.addEventListener('play', refreshAllPlayUI);
audio.addEventListener('pause', refreshAllPlayUI);
audio.addEventListener('loadedmetadata', updateEpisodeModalControls);
audio.addEventListener('timeupdate', () => {
  if (detailIndex === currentIndex) updateEpisodeModalControls();
});
audio.addEventListener('ended', () => {
  // Auto-play the previous episode released before the one that just finished
  // (episodes[] is newest -> oldest, so "older" = higher index)
  const olderIndex = currentIndex + 1;
  if (olderIndex < episodes.length) {
    playEpisode(olderIndex);
  } else {
    stopPlayback();
  }
});

function refreshAllPlayUI() {
  updateMiniPlayer();
  updateEpisodeModalControls();
  renderEpisodeList();
  updateMiniPlayerVisibility();
  if ('mediaSession' in navigator && currentIndex >= 0) {
    navigator.mediaSession.playbackState = audio.paused ? 'paused' : 'playing';
  }
}

// ============================================================
// MINI PLAYER
// ============================================================
function updateMiniPlayer() {
  if (currentIndex < 0) {
    miniPlayer.classList.remove('show');
    return;
  }
  const ep = episodes[currentIndex];
  miniPlayerArt.src = ep.image;
  miniPlayerTitle.textContent = ep.title;
  miniPlayerPlayPauseIcon.textContent = audio.paused ? 'play_arrow' : 'pause';
  updateMiniPlayerVisibility();
}

function updateMiniPlayerVisibility() {
  const hasEpisode = currentIndex >= 0;
  const anyModalOpen = !!document.querySelector('.kll-modal-overlay.show');
  const signedOut = authScreenEl && authScreenEl.style.display !== 'none';
  miniPlayer.classList.toggle('show', hasEpisode && !anyModalOpen && !signedOut);
}
// Modals can open/close from many places in main.js, so poll rather than
// hook every call site.
setInterval(updateMiniPlayerVisibility, 300);

miniPlayerPlayPause.addEventListener('click', (e) => {
  e.stopPropagation();
  audio.paused ? resumePlayback() : pausePlayback();
});

miniPlayerClose.addEventListener('click', (e) => {
  e.stopPropagation();
  stopPlayback();
});

miniPlayer.addEventListener('click', () => {
  if (currentIndex < 0) return;
  if (listenNavBtn) listenNavBtn.click(); // this alone triggers loadEpisodes() via its own listener
  openEpisodeDetail(currentIndex);
});

// ============================================================
// MEDIA SESSION (lock screen / Now Playing)
// ============================================================
function updateMediaSession(ep) {
  if (!('mediaSession' in navigator)) return;
  navigator.mediaSession.metadata = new MediaMetadata({
    title: ep.title,
    artist: 'Kids Learning Lab',
    album: 'Kids Learning Lab',
    artwork: [
      { src: ep.image, sizes: '512x512', type: 'image/jpeg' },
      { src: ep.image, sizes: '256x256', type: 'image/jpeg' }
    ]
  });
  navigator.mediaSession.playbackState = 'playing';
  navigator.mediaSession.setActionHandler('play', resumePlayback);
  navigator.mediaSession.setActionHandler('pause', pausePlayback);
  navigator.mediaSession.setActionHandler('previoustrack', () => {
    if (currentIndex > 0) { playEpisode(currentIndex - 1); detailIndex = currentIndex; } // newer
  });
  navigator.mediaSession.setActionHandler('nexttrack', () => {
    if (currentIndex < episodes.length - 1) { playEpisode(currentIndex + 1); detailIndex = currentIndex; } // older
  });
}

// ============================================================
// INIT
// ============================================================
if (listenNavBtn) {
  listenNavBtn.addEventListener('click', () => {
    loadEpisodes();
  });
}