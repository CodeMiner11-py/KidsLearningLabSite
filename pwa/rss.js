// rss.js — Fetches and parses the Kids Learning Lab RSS feed

const RSS_URL = 'https://go.kidslearninglab.com/rss';
const CORS_PROXY = 'https://api.allorigins.win/get?url=';
const CACHE_KEY = 'kll-episodes-cache';
const CACHE_TTL = 30 * 60 * 1000; // 30 minutes

async function fetchRSS() {
  // Check localStorage cache
  try {
    const cached = localStorage.getItem(CACHE_KEY);
    if (cached) {
      const { timestamp, data } = JSON.parse(cached);
      if (Date.now() - timestamp < CACHE_TTL) return data;
    }
  } catch {}

  const res = await fetch(`${CORS_PROXY}${encodeURIComponent(RSS_URL)}`);
  const json = await res.json();
  const xml = new DOMParser().parseFromString(json.contents, 'text/xml');
  const episodes = parseEpisodes(xml);

  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify({ timestamp: Date.now(), data: episodes }));
  } catch {}

  return episodes;
}

function parseEpisodes(xml) {
  const channel = xml.querySelector('channel');
  const showArt = channel?.querySelector('image url')?.textContent ||
                  channel?.querySelector('itunes\\:image, image')?.getAttribute('href') || '';
  const items = [...xml.querySelectorAll('item')];

  return items.map((item, i) => {
    const get = tag => item.querySelector(tag)?.textContent?.trim() || '';
    const getAttr = (tag, attr) => item.querySelector(tag)?.getAttribute(attr) || '';

    // Episode artwork: itunes:image href, then channel art
    const art = item.querySelector('itunes\\:image')?.getAttribute('href') ||
                getAttr('image', 'href') || showArt;

    // Enclosure URL for in-app player
    const audioUrl = getAttr('enclosure', 'url') || '';

    // Duration
    const rawDuration = get('itunes\\:duration');
    const duration = formatDuration(rawDuration);

    // Guid as ID (slugified)
    const guid = get('guid') || `episode-${i}`;
    const id = slugify(get('title') || guid);

    // Spotify/Apple links from description
    const desc = get('description');

    // Season/episode numbers
    const season = get('itunes\\:season') || '';
    const epNum = get('itunes\\:episode') || '';

    // Explicit flag
    const explicit = get('itunes\\:explicit') === 'yes';

    return {
      id,
      guid,
      title: get('title'),
      description: stripHtml(desc),
      descriptionHtml: desc,
      pubDate: get('pubDate'),
      pubDateFormatted: formatDate(get('pubDate')),
      duration,
      art,
      audioUrl,
      season,
      epNum,
      explicit,
      link: get('link')
    };
  });
}

function formatDuration(raw) {
  if (!raw) return '';
  if (raw.includes(':')) return raw;
  const s = parseInt(raw, 10);
  if (isNaN(s)) return raw;
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${String(sec).padStart(2, '0')}`;
}

function formatDate(raw) {
  if (!raw) return '';
  try {
    return new Intl.DateTimeFormat('en-US', { year: 'numeric', month: 'short', day: 'numeric' }).format(new Date(raw));
  } catch { return raw; }
}

function stripHtml(html) {
  const div = document.createElement('div');
  div.innerHTML = html;
  return div.textContent?.trim() || '';
}

function slugify(str) {
  return str.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}

window.KLL = window.KLL || {};
window.KLL.fetchRSS = fetchRSS;
