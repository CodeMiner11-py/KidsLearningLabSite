/**
 * kll-player.js — shared podcast player engine for Kids Learning Lab.
 *
 * Provides:
 *   - A cached RSS episode fetcher (KLLPlayer.getEpisodes), 5-hour cache in
 *     localStorage so repeat visits within a window don't re-fetch/re-parse
 *     the whole feed.
 *   - A single shared <audio> element + play state (KLLPlayer state), so
 *     the full-screen modal player and the bottom-left miniplayer always
 *     agree on what's playing.
 *   - The full-screen modal player UI (openEpisodePlayer), lifted out of
 *     episodes.html so index.html and any future page can open the exact
 *     same player instead of duplicating ~250 lines of player markup.
 *   - The bottom-left miniplayer UI, which appears on any page that loads
 *     this script the moment something starts playing, and reopens the
 *     same full modal when clicked.
 *
 * IMPORTANT LIMITATION: this is a static multi-page site, not an SPA — a
 * real navigation between pages (index.html -> episodes.html) reloads the
 * whole document, which necessarily kills any in-page <audio> element and
 * its playback position. The miniplayer/modal here are the same shared
 * component on every page (so behavior is consistent), but audio does NOT
 * keep playing across a full page navigation. That would need a
 * persistent-audio architecture (e.g. a hidden iframe host + postMessage,
 * or a service worker) which is out of scope for this pass.
 */
(function (global) {
  'use strict';

  const RSS_URL      = 'https://anchor.fm/s/d1b8e6fc/podcast/rss';
  const FALLBACK_ART = 'https://kidslearninglab.com/wp-content/uploads/2025/02/podcast-logo-app-rounded.png';
  const CACHE_KEY     = 'kllEpisodeCache_v1';
  const CACHE_TTL_MS  = 5 * 60 * 60 * 1000; // 5 hours

  // ============================================================
  // RSS fetch + 5-hour cache
  // ============================================================
  function stripHtml(html) {
    const d = document.createElement('div');
    d.innerHTML = html;
    return d.textContent || d.innerText || '';
  }

  function parseRSS(text) {
    const blocks = [];
    const re = /<item>([\s\S]*?)<\/item>/g;
    let m;
    while ((m = re.exec(text)) !== null) blocks.push(m[1]);
    return blocks.map((block) => {
      const get = (tag) => {
        const cdataM = block.match(new RegExp(`<${tag}[^>]*><!\\[CDATA\\[([\\s\\S]*?)\\]\\]><\\/${tag}>`));
        if (cdataM) return cdataM[1].trim();
        const plainM = block.match(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`));
        return plainM ? plainM[1].trim() : '';
      };
      const encM = block.match(/<enclosure[^>]+url="([^"]+)"/);
      const artM = block.match(/<itunes:image[^>]+href="([^"]+)"/);
      const durM = block.match(/<itunes:duration[^>]*>([^<]+)<\/itunes:duration>/);
      return {
        title:    get('title'),
        pubDate:  get('pubDate'),
        desc:     stripHtml(get('description') || get('itunes:summary') || ''),
        audioUrl: encM ? encM[1] : null,
        art:      artM ? artM[1] : FALLBACK_ART,
        duration: durM ? durM[1].trim() : '',
      };
    }).filter((e) => e.audioUrl);
  }

  // Returns a Promise<episode[]>. Reads a valid (< 5h old) cache first;
  // otherwise fetches, parses, and re-caches. A stale/corrupt cache entry
  // is just treated as a miss rather than thrown — this should never block
  // the page from loading episodes.
  async function getEpisodes() {
    try {
      const raw = localStorage.getItem(CACHE_KEY);
      if (raw) {
        const cached = JSON.parse(raw);
        if (cached && Array.isArray(cached.episodes) && (Date.now() - cached.fetchedAt) < CACHE_TTL_MS) {
          return cached.episodes;
        }
      }
    } catch (e) { /* corrupt cache — fall through to a fresh fetch */ }

    const res  = await fetch(RSS_URL);
    const text = await res.text();
    const episodes = parseRSS(text);

    try {
      localStorage.setItem(CACHE_KEY, JSON.stringify({ fetchedAt: Date.now(), episodes }));
    } catch (e) { /* localStorage full/unavailable — playback still works, just uncached */ }

    return episodes;
  }

  function slugify(title) {
    return title
      .toLowerCase()
      .replace(/['’]/g, '')
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '');
  }

  function fmtTime(s) {
    if (!s || isNaN(s)) return '0:00';
    const m = Math.floor(s / 60), sec = Math.floor(s % 60);
    return m + ':' + (sec < 10 ? '0' : '') + sec;
  }

  function fmtDate(str) {
    if (!str) return '';
    const d = new Date(str);
    return isNaN(d) ? '' : d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  }

  // ============================================================
  // Shared playback state — one <audio> for the whole page, so the modal
  // and the miniplayer are always showing/controlling the same thing.
  // ============================================================
  const audio = new Audio();
  audio.preload = 'metadata';

  let episodeList  = [];  // whatever list "next/prev" should walk through
  let currentIndex = -1;
  let currentEp    = null;

  const listeners = []; // fires on any state change: play/pause/track change
  function notify() { listeners.forEach((fn) => fn(state())); }
  function onStateChange(fn) { listeners.push(fn); }

  function state() {
    return {
      episode: currentEp,
      playing: !audio.paused && !audio.ended,
      currentTime: audio.currentTime,
      duration: audio.duration,
    };
  }

  audio.addEventListener('play',  notify);
  audio.addEventListener('pause', notify);
  audio.addEventListener('timeupdate', notify);
  audio.addEventListener('loadedmetadata', notify);
  audio.addEventListener('ended', () => {
    notify();
    if (currentIndex + 1 < episodeList.length) playEpisode(episodeList[currentIndex + 1], episodeList);
  });

  // list (optional): the array this episode came from, so next/prev work
  // inside the modal. Defaults to just [episode] if not given.
  function playEpisode(episode, list) {
    episodeList  = list || [episode];
    currentIndex = episodeList.findIndex((e) => e.audioUrl === episode.audioUrl);
    if (currentIndex === -1) currentIndex = 0;
    currentEp = episode;

    if (audio.src !== episode.audioUrl) {
      audio.src = episode.audioUrl;
    }
    audio.play().catch(() => { /* autoplay blocked — user can hit play manually */ });
    notify();
  }

  function togglePlay() {
    if (!currentEp) return;
    if (audio.paused) audio.play(); else audio.pause();
  }

  function playAdjacent(dir) {
    const next = currentIndex + dir;
    if (next >= 0 && next < episodeList.length) playEpisode(episodeList[next], episodeList);
  }

  // ============================================================
  // Full-screen modal player (shared markup, injected once)
  // ============================================================
  let modalOverlay, modalInner;

  function ensureModalMarkup() {
    if (modalOverlay) return;
    modalOverlay = document.createElement('div');
    modalOverlay.className = 'kllp-modal-overlay';
    modalOverlay.id = 'kllp-modal-overlay';
    modalOverlay.setAttribute('role', 'dialog');
    modalOverlay.setAttribute('aria-modal', 'true');
    modalOverlay.innerHTML = `
      <div class="kllp-modal" id="kllp-modal">
        <button class="kllp-modal-close" id="kllp-modal-close" aria-label="Close player">✕</button>
        <div id="kllp-modal-inner"></div>
      </div>`;
    document.body.appendChild(modalOverlay);
    modalInner = document.getElementById('kllp-modal-inner');

    document.getElementById('kllp-modal-close').addEventListener('click', closeModal);
    modalOverlay.addEventListener('click', (e) => { if (e.target === modalOverlay) closeModal(); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && modalOverlay.classList.contains('open')) closeModal(); });
  }

  function renderModalBody() {
    const ep = currentEp;
    if (!ep || !modalInner) return;

    const shareUrl  = encodeURIComponent(window.location.origin + '/episodes.html?ep=' + slugify(ep.title));
    const shareText = encodeURIComponent('🎧 Check out this episode: ' + ep.title);

    modalInner.innerHTML = `
      <img class="kllp-modal-art" src="${ep.art}" alt="${ep.title}" onerror="this.src='${FALLBACK_ART}'">
      <div class="kllp-modal-body">
        <div class="kllp-modal-label">Now Playing</div>
        <div class="kllp-modal-title">${ep.title}</div>
        <div class="kllp-modal-meta">${fmtDate(ep.pubDate)}</div>
        <div class="kllp-modal-desc">${ep.desc}</div>

        <div class="kllp-progress-wrap">
          <div class="kllp-progress-bg"><div class="kllp-progress-fill" id="kllp-fill"></div></div>
          <input type="range" class="kllp-seek" id="kllp-seek" min="0" max="100" value="0" step="0.1">
        </div>
        <div class="kllp-times"><span id="kllp-cur">0:00</span><span id="kllp-dur">--:--</span></div>

        <div class="kllp-controls">
          <button class="kllp-btn" id="kllp-prev" title="Previous episode">
            <svg viewBox="0 0 24 24"><path d="M6 6h2v12H6zm3.5 6 8.5 6V6z"/></svg>
          </button>
          <button class="kllp-btn" id="kllp-back" title="−15s">
            <svg viewBox="0 0 24 24"><path d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z"/><text x="7" y="15.5" font-size="5.5" font-family="Albert Sans,sans-serif" font-weight="800" fill="currentColor">15</text></svg>
          </button>
          <button class="kllp-btn play-pause" id="kllp-play" title="Play/Pause">
            <svg id="kllp-play-icon" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
          </button>
          <button class="kllp-btn" id="kllp-fwd" title="+15s">
            <svg viewBox="0 0 24 24"><path d="M12 5V1l5 5-5 5V7c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6h2c0 4.42-3.58 8-8 8s-8-3.58-8-8 3.58-8 8-8z"/><text x="7" y="15.5" font-size="5.5" font-family="Albert Sans,sans-serif" font-weight="800" fill="currentColor">15</text></svg>
          </button>
          <button class="kllp-btn" id="kllp-next" title="Next episode">
            <svg viewBox="0 0 24 24"><path d="M6 18l8.5-6L6 6v12zm2.5-6 6-4.35V16.35L8.5 12zM16 6h2v12h-2z"/></svg>
          </button>
          <div class="kllp-vol-wrap">
            <svg viewBox="0 0 24 24"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/></svg>
            <input type="range" class="kllp-vol" id="kllp-vol" min="0" max="1" step="0.02" value="${audio.volume}">
          </div>
        </div>

        <div class="kllp-share-row">
          <span class="kllp-share-label">Share</span>
          <button class="kllp-share-btn copy-btn" id="kllp-copy-btn">
            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>
            <span>Copy link</span>
          </button>
          <a class="kllp-share-btn" href="https://twitter.com/intent/tweet?text=${shareText}&url=${shareUrl}" target="_blank" rel="noopener">
            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231-5.401 6.231H2.746l7.73-8.835L1.254 2.25H8.08l4.253 5.622 5.912-5.622zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
            <span>X / Twitter</span>
          </a>
          <a class="kllp-share-btn" href="https://www.facebook.com/sharer/sharer.php?u=${shareUrl}" target="_blank" rel="noopener">
            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/></svg>
            <span>Facebook</span>
          </a>
          <a class="kllp-share-btn" href="https://wa.me/?text=${shareText}%20${shareUrl}" target="_blank" rel="noopener">
            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51a12.8 12.8 0 0 0-.57-.01c-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 0 1-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 0 1-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 0 1 2.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0 0 12.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 0 0 5.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 0 0-3.48-8.413z"/></svg>
            <span>WhatsApp</span>
          </a>
        </div>
      </div>`;

    wireModalControls();
  }

  function wireModalControls() {
    const fill     = document.getElementById('kllp-fill');
    const seek     = document.getElementById('kllp-seek');
    const curEl    = document.getElementById('kllp-cur');
    const durEl    = document.getElementById('kllp-dur');
    const playBtn  = document.getElementById('kllp-play');
    const playIcon = document.getElementById('kllp-play-icon');
    const vol      = document.getElementById('kllp-vol');
    const copyBtn  = document.getElementById('kllp-copy-btn');

    const PLAY  = 'M8 5v14l11-7z';
    const PAUSE = 'M6 19h4V5H6v14zm8-14v14h4V5h-4z';
    function setIcon(playing) { playIcon.querySelector('path').setAttribute('d', playing ? PAUSE : PLAY); }

    function syncFromAudio() {
      durEl.textContent = fmtTime(audio.duration);
      seek.max = audio.duration || 0;
      const pct = audio.duration ? (audio.currentTime / audio.duration) * 100 : 0;
      fill.style.width = pct + '%';
      seek.value = audio.currentTime;
      curEl.textContent = fmtTime(audio.currentTime);
      setIcon(!audio.paused && !audio.ended);
    }
    syncFromAudio();
    onStateChange(syncFromAudio); // keep the open modal in sync if state changes elsewhere (e.g. miniplayer)

    playBtn.addEventListener('click', togglePlay);
    seek.addEventListener('input', () => { audio.currentTime = seek.value; });
    document.getElementById('kllp-back').addEventListener('click', () => { audio.currentTime = Math.max(0, audio.currentTime - 15); });
    document.getElementById('kllp-fwd').addEventListener('click',  () => { audio.currentTime = Math.min(audio.duration || 0, audio.currentTime + 15); });
    document.getElementById('kllp-prev').addEventListener('click', () => { playAdjacent(-1); renderModalBody(); });
    document.getElementById('kllp-next').addEventListener('click', () => { playAdjacent(1); renderModalBody(); });
    vol.addEventListener('input', () => { audio.volume = vol.value; });
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(window.location.origin + '/episodes.html?ep=' + slugify(currentEp.title)).then(() => {
        copyBtn.classList.add('copied');
        copyBtn.querySelector('span').textContent = 'Copied!';
        setTimeout(() => { copyBtn.classList.remove('copied'); copyBtn.querySelector('span').textContent = 'Copy link'; }, 2000);
      });
    });
  }

  function openModal() {
    ensureModalMarkup();
    if (!currentEp) return;
    renderModalBody();
    modalOverlay.classList.add('open');
    document.body.style.overflow = 'hidden';
  }

  function closeModal() {
    if (!modalOverlay) return;
    modalOverlay.classList.remove('open');
    document.body.style.overflow = '';
    // Playback deliberately continues — closing the modal is like
    // backgrounding a podcast app, not stopping it. The miniplayer picks
    // up showing what's still playing.
  }

  // Opens the modal already playing `episode`. `list` (optional) is the
  // array to walk with next/prev — pass the full episode array from a
  // carousel/list so those controls work; omit for a single stray link.
  function openEpisodePlayer(episode, list) {
    playEpisode(episode, list);
    openModal();
  }

  // ============================================================
  // Bottom-left miniplayer
  // ============================================================
  let miniEl;

  function ensureMiniMarkup() {
    if (miniEl) return;
    miniEl = document.createElement('button');
    miniEl.type = 'button';
    miniEl.className = 'kllp-mini';
    miniEl.id = 'kllp-mini';
    miniEl.setAttribute('aria-label', 'Open player');
    miniEl.innerHTML = `
      <img class="kllp-mini-art" id="kllp-mini-art" src="" alt="">
      <div class="kllp-mini-eq" id="kllp-mini-eq"><span></span><span></span><span></span></div>
    `;
    miniEl.addEventListener('click', openModal);
    document.body.appendChild(miniEl);
  }

  function renderMini() {
    if (!currentEp) return; // never shown until something has actually played
    ensureMiniMarkup();
    document.getElementById('kllp-mini-art').src = currentEp.art || FALLBACK_ART;
    miniEl.classList.add('show');
    miniEl.classList.toggle('playing', !audio.paused && !audio.ended);
  }

  onStateChange(renderMini);

  // ============================================================
  // Public API
  // ============================================================
  global.KLLPlayer = {
    getEpisodes,
    openEpisodePlayer,
    playEpisode,
    togglePlay,
    slugify,
    fmtDate,
    fmtTime,
    FALLBACK_ART,
    onStateChange,
    state,
  };
})(window);
