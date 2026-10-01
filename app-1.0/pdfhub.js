// ============================================================
// FILE HUB (file is still called pdfhub.js)
// Two kinds of file, both stored as plain text:
//   * PDF   -> extract plaintext on-device (pdf.js, lazy-loaded from cdnjs)
//   * Image -> downscaled on-device, sent once to the image-scan worker where
//              Qwen reads it and returns { title, alltext, diagrams }. The
//              image itself is NEVER stored; only the text Qwen returns is.
// Text is whitespace-collapsed and stored LOCALLY in IndexedDB (nothing is
// uploaded to Firebase). Only that plaintext is ever sent anywhere, and only
// when generating a course/game.
// Used by:
//   * Profile > File Hub page (upload / rename / delete)
//   * Create Course modal + every Learning Game chooser (mountPdfPicker)
//
// IndexedDB "kll-pdfs" / store "pdfs":
//   { id, uid, name, chars, pages, kind: 'pdf' | 'image', createdAt, text }
//   (rows saved before images existed have no `kind` and are PDFs)
// ============================================================
import { auth } from './firebase.js';
import { onAuthStateChanged } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";

const PDFJS_VERSION = '3.11.174';
const PDFJS_SRC = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${PDFJS_VERSION}/pdf.min.js`;
const PDFJS_WORKER_SRC = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${PDFJS_VERSION}/pdf.worker.min.js`;

// Deployed image-scan worker (imagescan-worker.js). POST { image: "data:image/jpeg;base64,..." }
// -> { title, alltext, diagrams }.
const IMAGE_SCAN_WORKER_URL = 'https://imagescanworker.nameless-cherry-998c.workers.dev'; // TODO: paste your deployed image-scan worker URL

const MAX_FILE_BYTES = 25 * 1024 * 1024; // 25 MB upload cap
const MAX_IMAGE_BYTES = 25 * 1024 * 1024; // photos are shrunk before sending, so this only guards absurd files
const IMAGE_MAX_SIDE = 1280;              // long edge after downscale (still readable for text/diagrams)
const IMAGE_MIN_SIDE = 960;              // don't shrink below this just to hit the size target
const IMAGE_QUALITIES = [0.72, 0.6, 0.5]; // JPEG qualities tried at each size, best first
const IMAGE_TARGET_DATAURL_CHARS = 600_000; // ~450 KB: what we aim to send
const IMAGE_MAX_DATAURL_CHARS = 3_800_000;  // hard ceiling — stay under Groq's 4 MB base64 limit
const MAX_STORED_CHARS = 250000;         // plenty for AI prompts; keeps storage small
export const MAX_PDFS = 30; // max files of any kind

// ---- state ----
let pdfList = null;            // [{ id, name, chars, pages }] newest first, or null = not loaded
const textCache = new Map();   // id -> text
const pickers = new Set();     // mounted pickers, re-rendered when the list changes
let pdfjsPromise = null;

const uid = () => auth.currentUser?.uid || null;

onAuthStateChanged(auth, (user) => {
  pdfList = null;
  textCache.clear();
  pickers.forEach((p) => p.reset());
  renderHub();
  if (user) listPdfs(true).then(() => notifyChanged()).catch(() => {});
});

// ------------------------------------------------------------
// pdf.js loading (UMD build via <script>, worker via blob URL so it
// also works inside Capacitor's WKWebView, which blocks cross-origin workers)
// ------------------------------------------------------------
function loadPdfJs() {
  if (window.pdfjsLib) return Promise.resolve(window.pdfjsLib);
  if (pdfjsPromise) return pdfjsPromise;
  pdfjsPromise = new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = PDFJS_SRC;
    s.onload = async () => {
      try {
        const lib = window.pdfjsLib;
        const res = await fetch(PDFJS_WORKER_SRC);
        const blob = new Blob([await res.text()], { type: 'text/javascript' });
        lib.GlobalWorkerOptions.workerSrc = URL.createObjectURL(blob);
        resolve(lib);
      } catch (err) { reject(err); }
    };
    s.onerror = () => reject(new Error('Could not load the PDF reader. Check your connection.'));
    document.head.appendChild(s);
  }).catch((err) => { pdfjsPromise = null; throw err; });
  return pdfjsPromise;
}

// Removes every newline / blank run so the stored text is as small as possible.
export function compactText(raw) {
  return String(raw || '')
    .replace(/\u0000/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

async function extractPdfText(file) {
  const lib = await loadPdfJs();
  const data = new Uint8Array(await file.arrayBuffer());
  const pdf = await lib.getDocument({ data }).promise;
  let text = '';
  for (let p = 1; p <= pdf.numPages; p++) {
    const page = await pdf.getPage(p);
    const content = await page.getTextContent();
    text += content.items.map((it) => it.str).join(' ') + ' ';
    if (text.length > MAX_STORED_CHARS * 1.3) break; // enough; stop reading pages
  }
  const pages = pdf.numPages;
  pdf.destroy?.();
  text = compactText(text).slice(0, MAX_STORED_CHARS);
  return { text, pages };
}

// ------------------------------------------------------------
// Image files: shrink on-device -> Qwen reads it (worker) -> keep only the text
// ------------------------------------------------------------
async function loadImageBitmapFrom(file) {
  if (window.createImageBitmap) {
    try { return await createImageBitmap(file); } catch { /* fall through to <img> */ }
  }
  const url = URL.createObjectURL(file);
  try {
    return await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('Could not open that image.'));
      img.src = url;
    });
  } finally { URL.revokeObjectURL(url); }
}

// Compresses the photo on-device before it is sent: downscales to IMAGE_MAX_SIDE on the long
// edge, re-encodes as JPEG, and steps quality (then size) down until it is under
// IMAGE_TARGET_DATAURL_CHARS. If it can't reach the target without dropping below
// IMAGE_MIN_SIDE, the smallest version that still fits the hard ceiling is used.
async function prepareImageDataUrl(file) {
  const bmp = await loadImageBitmapFrom(file);
  const w0 = bmp.width || bmp.naturalWidth;
  const h0 = bmp.height || bmp.naturalHeight;
  if (!w0 || !h0) throw new Error('Could not open that image.');
  const longEdge = Math.max(w0, h0);

  const draw = (scale) => {
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(w0 * scale));
    canvas.height = Math.max(1, Math.round(h0 * scale));
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#fff'; // transparent PNGs would otherwise turn black
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
    return canvas;
  };

  try {
    const fit = Math.min(1, IMAGE_MAX_SIDE / longEdge);
    const floor = Math.min(fit, IMAGE_MIN_SIDE / longEdge);
    let scale = fit;
    let smallest = null;

    // Phase 1: aim for the target size, without going below IMAGE_MIN_SIDE.
    for (let attempt = 0; attempt < 8; attempt++) {
      const canvas = draw(scale);
      for (const q of IMAGE_QUALITIES) {
        const dataUrl = canvas.toDataURL('image/jpeg', q);
        if (dataUrl.length <= IMAGE_TARGET_DATAURL_CHARS) return dataUrl;
        smallest = dataUrl; // qualities run high -> low, so this is the smallest at this size
      }
      if (scale <= floor) break;
      scale = Math.max(floor, scale * 0.85);
    }
    if (smallest && smallest.length <= IMAGE_MAX_DATAURL_CHARS) return smallest;

    // Phase 2 (rare): still over the hard ceiling, so shrink further regardless of size.
    scale = floor;
    for (let attempt = 0; attempt < 4; attempt++) {
      scale *= 0.75;
      const dataUrl = draw(scale).toDataURL('image/jpeg', IMAGE_QUALITIES[IMAGE_QUALITIES.length - 1]);
      if (dataUrl.length <= IMAGE_MAX_DATAURL_CHARS) return dataUrl;
    }
    throw new Error('That image is too big to scan. Try a smaller one.');
  } finally {
    bmp.close?.();
  }
}

// At most 5 words, no quotes/punctuation — the name shown in the File Hub.
function shortTitle(raw, fallback) {
  const words = String(raw || '').replace(/["'`*_#:.,;!?()[\]{}]/g, ' ').split(/\s+/).filter(Boolean).slice(0, 5);
  return (words.join(' ') || fallback).slice(0, 60);
}

// Asks the worker (Qwen) to read the image. Returns { title, text } where `text` is ONLY what Qwen
// returned: alltext, then one "Diagram: …" sentence per diagram.
async function scanImage(file) {
  if (!IMAGE_SCAN_WORKER_URL) throw new Error('Image scanning is not set up yet.');
  const image = await prepareImageDataUrl(file);
  let res;
  try {
    res = await fetch(IMAGE_SCAN_WORKER_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image }),
    });
  } catch {
    throw new Error('Could not reach the image scanner. Check your connection.');
  }
  let data = null;
  try { data = await res.json(); } catch { /* handled below */ }
  if (!res.ok || !data || data.error) throw new Error(data?.error || 'Could not read that image. Try again.');

  const parts = [];
  if (typeof data.alltext === 'string') parts.push(data.alltext);
  if (data.diagrams && typeof data.diagrams === 'object') {
    for (const d of Object.values(data.diagrams)) {
      if (typeof d === 'string' && d.trim()) parts.push(`Diagram: ${d.trim()}`);
    }
  }
  return { title: data.title, text: compactText(parts.join(' ')).slice(0, MAX_STORED_CHARS) };
}

// ------------------------------------------------------------
// Data layer (IndexedDB, per signed-in user)
// ------------------------------------------------------------
let dbPromise = null;
function openDb() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open('kll-pdfs', 1);
    req.onupgradeneeded = () => {
      const store = req.result.createObjectStore('pdfs', { keyPath: 'id' });
      store.createIndex('uid', 'uid');
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => { dbPromise = null; reject(req.error); };
  });
  return dbPromise;
}

async function tx(mode, fn) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const t = db.transaction('pdfs', mode);
    const out = fn(t.objectStore('pdfs'));
    t.oncomplete = () => resolve(out?.result);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}

const reqP = (r) => new Promise((res, rej) => { r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });

export async function listPdfs(force = false) {
  const u = uid();
  if (!u) return [];
  if (pdfList && !force) return pdfList;
  const db = await openDb();
  const rows = await reqP(db.transaction('pdfs').objectStore('pdfs').index('uid').getAll(u));
  pdfList = rows
    .sort((x, y) => (y.createdAt || 0) - (x.createdAt || 0))
    .map(({ text, ...meta }) => meta);
  return pdfList;
}

// Returns the stored PLAINTEXT string (already whitespace-collapsed).
export async function getPdfText(id) {
  if (textCache.has(id)) return textCache.get(id);
  const u = uid();
  if (!u || !id) return '';
  const db = await openDb();
  const row = await reqP(db.transaction('pdfs').objectStore('pdfs').get(id));
  const text = row && row.uid === u ? (row.text || '') : '';
  textCache.set(id, text);
  return text;
}

export async function uploadPdf(file) {
  const u = uid();
  if (!u) throw new Error('Sign in first.');
  if (!file) throw new Error('No file chosen.');
  if (file.type && file.type !== 'application/pdf' && !/\.pdf$/i.test(file.name)) {
    throw new Error('That is not a PDF.');
  }
  if (file.size > MAX_FILE_BYTES) throw new Error('That PDF is over 25 MB.');
  const list = await listPdfs();
  if (list.length >= MAX_PDFS) throw new Error(`You can keep up to ${MAX_PDFS} files. Delete one first.`);

  const { text, pages } = await extractPdfText(file);
  if (text.length < 20) throw new Error('No readable text found. Scanned PDFs are not supported.');

  const id = `pdf_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  const name = file.name.replace(/\.pdf$/i, '').trim().slice(0, 60) || 'Untitled PDF';
  const meta = { id, uid: u, name, chars: text.length, pages, kind: 'pdf', createdAt: Date.now() };
  await tx('readwrite', (st) => st.put({ ...meta, text }));

  textCache.set(id, text);
  pdfList = [meta, ...(pdfList || [])];
  notifyChanged();
  return meta;
}

export async function uploadImage(file) {
  const u = uid();
  if (!u) throw new Error('Sign in first.');
  if (!file) throw new Error('No file chosen.');
  if (file.type && !file.type.startsWith('image/')) throw new Error('That is not an image.');
  if (file.size > MAX_IMAGE_BYTES) throw new Error('That image is over 25 MB.');
  const list = await listPdfs();
  if (list.length >= MAX_PDFS) throw new Error(`You can keep up to ${MAX_PDFS} files. Delete one first.`);

  const { title, text } = await scanImage(file);
  if (text.length < 20) throw new Error('No readable text or diagrams found in that image.');

  const id = `img_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  const name = shortTitle(title, 'Photo notes');
  const meta = { id, uid: u, name, chars: text.length, pages: 0, kind: 'image', createdAt: Date.now() };
  await tx('readwrite', (st) => st.put({ ...meta, text }));

  textCache.set(id, text);
  pdfList = [meta, ...(pdfList || [])];
  notifyChanged();
  return meta;
}

export async function renamePdf(id, newName) {
  const u = uid();
  const name = String(newName || '').trim().slice(0, 60);
  if (!u || !name) return;
  const db = await openDb();
  const row = await reqP(db.transaction('pdfs').objectStore('pdfs').get(id));
  if (!row || row.uid !== u) return;
  await tx('readwrite', (st) => st.put({ ...row, name }));
  pdfList = (pdfList || []).map((p) => (p.id === id ? { ...p, name } : p));
  notifyChanged();
}

export async function deletePdf(id) {
  const u = uid();
  if (!u) return;
  await tx('readwrite', (st) => st.delete(id));
  textCache.delete(id);
  pdfList = (pdfList || []).filter((p) => p.id !== id);
  notifyChanged();
}

// Picks a slice of a long text to send to the AI. `fraction` (0..1) chooses
// where in the document the window starts, so unit 1 reads the start and
// unit 10 reads the end. Whole text is returned when it already fits.
export function excerptText(text, max = 24000, fraction = 0) {
  if (!text || text.length <= max) return text || '';
  const start = Math.max(0, Math.min(text.length - max, Math.floor((text.length - max) * fraction)));
  return text.slice(start, start + max);
}

// ------------------------------------------------------------
// File Hub page (Profile > File Hub)
// ------------------------------------------------------------
const hubBtn = document.getElementById('profilePdfHubBtn');
const hubOverlay = document.getElementById('pdfHubPageOverlay');
const hubExitBtn = document.getElementById('pdfHubExitBtn');
const hubUploadBtn = document.getElementById('pdfHubUploadBtn');
const hubFileInput = document.getElementById('pdfHubFileInput');
const hubImageBtn = document.getElementById('pdfHubImageBtn');
const hubImageInput = document.getElementById('pdfHubImageInput');
const hubList = document.getElementById('pdfHubList');
const hubStatus = document.getElementById('pdfHubStatus');
const hubCountLabel = document.getElementById('profilePdfHubCount');
const renameOverlay = document.getElementById('pdfRenameOverlay');
const renameInput = document.getElementById('pdfRenameInput');
const renameSaveBtn = document.getElementById('pdfRenameSaveBtn');
const renameCancelBtn = document.getElementById('pdfRenameCancelBtn');

let renameId = null;
let confirmDeleteId = null;
let confirmTimer = null;

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function fmtChars(n) {
  return n >= 1000 ? `${Math.round(n / 1000)}k characters` : `${n} characters`;
}

function renderHub() {
  if (!hubList) return;
  const list = pdfList || [];
  if (hubCountLabel) hubCountLabel.textContent = list.length ? `${list.length} file${list.length === 1 ? '' : 's'}` : 'Upload files to learn from';
  if (!list.length) {
    hubList.innerHTML = '<div class="pdf-hub-empty">No files yet. Upload a PDF or an image to build courses and games from it.</div>';
    return;
  }
  hubList.innerHTML = list.map((p) => `
    <div class="pdf-hub-row" data-id="${p.id}">
      <div class="pdf-hub-icon"><span class="material-symbols-outlined">${p.kind === 'image' ? 'image' : 'picture_as_pdf'}</span></div>
      <div class="pdf-hub-info">
        <div class="pdf-hub-name">${esc(p.name)}</div>
        <div class="pdf-hub-meta">${p.kind === 'image' ? 'Image · ' : (p.pages ? `${p.pages} page${p.pages === 1 ? '' : 's'} · ` : '')}${fmtChars(p.chars || 0)}</div>
      </div>
      <button type="button" class="pdf-hub-icon-btn" data-act="rename" aria-label="Rename"><span class="material-symbols-outlined">edit</span></button>
      <button type="button" class="pdf-hub-icon-btn ${confirmDeleteId === p.id ? 'confirming' : ''}" data-act="delete" aria-label="Delete"><span class="material-symbols-outlined">${confirmDeleteId === p.id ? 'check' : 'delete'}</span></button>
    </div>`).join('');
}

async function openHub() {
  hubStatus.textContent = '';
  hubOverlay.classList.add('show');
  try {
    await listPdfs(true);
  } catch (err) {
    console.warn('File list failed:', err);
    hubStatus.textContent = 'Could not load your files.';
  }
  renderHub();
}

async function handleUpload(file, { statusEl, onDone, kind = 'pdf' } = {}) {
  const status = statusEl || hubStatus;
  const isImage = kind === 'image';
  status.classList.remove('error');
  status.textContent = isImage ? 'Scanning image…' : 'Reading PDF…';
  try {
    const meta = await (isImage ? uploadImage(file) : uploadPdf(file));
    status.textContent = '';
    onDone?.(meta);
    return meta;
  } catch (err) {
    console.error(`${isImage ? 'Image' : 'PDF'} upload failed:`, err);
    status.classList.add('error');
    status.textContent = err.message || (isImage ? 'Could not read that image.' : 'Could not read that PDF.');
    return null;
  }
}

if (hubBtn) {
  hubBtn.addEventListener('click', openHub);
  hubExitBtn.addEventListener('click', () => hubOverlay.classList.remove('show'));

  hubUploadBtn.addEventListener('click', () => hubFileInput.click());
  hubFileInput.addEventListener('change', async () => {
    const file = hubFileInput.files?.[0];
    hubFileInput.value = '';
    if (!file) return;
    hubUploadBtn.disabled = true;
    if (hubImageBtn) hubImageBtn.disabled = true;
    await handleUpload(file);
    hubUploadBtn.disabled = false;
    if (hubImageBtn) hubImageBtn.disabled = false;
  });

  hubImageBtn?.addEventListener('click', () => hubImageInput.click());
  hubImageInput?.addEventListener('change', async () => {
    const file = hubImageInput.files?.[0];
    hubImageInput.value = '';
    if (!file) return;
    hubUploadBtn.disabled = true;
    hubImageBtn.disabled = true;
    await handleUpload(file, { kind: 'image' });
    hubUploadBtn.disabled = false;
    hubImageBtn.disabled = false;
  });

  hubList.addEventListener('click', async (e) => {
    const btn = e.target.closest('[data-act]');
    if (!btn) return;
    const id = btn.closest('.pdf-hub-row').dataset.id;
    if (btn.dataset.act === 'rename') {
      const p = (pdfList || []).find((x) => x.id === id);
      renameId = id;
      renameInput.value = p?.name || '';
      renameOverlay.classList.add('show');
      renameInput.focus();
      return;
    }
    // Delete: first tap arms it (turns red), second tap within 3s confirms.
    if (confirmDeleteId !== id) {
      confirmDeleteId = id;
      clearTimeout(confirmTimer);
      confirmTimer = setTimeout(() => { confirmDeleteId = null; renderHub(); }, 3000);
      renderHub();
      return;
    }
    clearTimeout(confirmTimer);
    confirmDeleteId = null;
    try { await deletePdf(id); } catch (err) { hubStatus.classList.add('error'); hubStatus.textContent = 'Could not delete that file.'; }
    renderHub();
  });

  renameCancelBtn.addEventListener('click', () => renameOverlay.classList.remove('show'));
  renameSaveBtn.addEventListener('click', async () => {
    const id = renameId;
    renameOverlay.classList.remove('show');
    if (!id) return;
    try { await renamePdf(id, renameInput.value); } catch (err) { hubStatus.classList.add('error'); hubStatus.textContent = 'Could not rename that file.'; }
    renderHub();
  });
  renameInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') renameSaveBtn.click(); });
}

function notifyChanged() {
  renderHub();
  pickers.forEach((p) => p.render());
}

// ------------------------------------------------------------
// Reusable picker — drop into any modal:
//   const picker = mountPdfPicker(document.getElementById('xPdfPicker'));
//   picker.getSelected()  -> { id, name } | null
//   picker.reset()        -> back to "No file"
// ------------------------------------------------------------
export function mountPdfPicker(container, { onChange } = {}) {
  if (!container) return { getSelected: () => null, reset() {}, render() {} };
  let selectedId = '';
  let busy = false;

  container.classList.add('pdf-picker');
  container.innerHTML = `
    <div class="pdf-picker-title"><span class="material-symbols-outlined">folder</span> File Hub: Use a file instead</div>
    <div class="pdf-picker-row">
      <select class="pdf-picker-select" aria-label="Choose a file"></select>
    </div>
    <div class="pdf-picker-row pdf-picker-uploads">
      <button type="button" class="pdf-picker-upload pdf-picker-upload-pdf">Upload PDF</button>
      <button type="button" class="pdf-picker-upload pdf-picker-upload-img">Upload Image</button>
    </div>
    <input type="file" accept="application/pdf,.pdf" class="pdf-picker-file" hidden>
    <input type="file" accept="image/*" class="pdf-picker-imgfile" hidden>
    <div class="pdf-picker-status"></div>`;
  const select = container.querySelector('.pdf-picker-select');
  const uploadBtn = container.querySelector('.pdf-picker-upload-pdf');
  const imageBtn = container.querySelector('.pdf-picker-upload-img');
  const fileInput = container.querySelector('.pdf-picker-file');
  const imageInput = container.querySelector('.pdf-picker-imgfile');
  const status = container.querySelector('.pdf-picker-status');

  const api = {
    getSelected() {
      const p = (pdfList || []).find((x) => x.id === selectedId);
      return p ? { id: p.id, name: p.name } : null;
    },
    reset() { selectedId = ''; api.render(); },
    render() {
      const list = pdfList || [];
      select.innerHTML = '<option value="">No file</option>' +
        list.map((p) => `<option value="${p.id}">${esc(p.name)}</option>`).join('');
      if (!list.some((p) => p.id === selectedId)) selectedId = '';
      select.value = selectedId;
      container.classList.toggle('has-selection', !!selectedId);
    },
  };

  select.addEventListener('change', () => {
    selectedId = select.value;
    container.classList.toggle('has-selection', !!selectedId);
    onChange?.(api.getSelected());
  });

  // Both upload buttons share one flow; `kind` picks PDF reading vs. image scanning.
  const wireUpload = (btn, input, kind) => {
    btn.addEventListener('click', () => { if (!busy) input.click(); });
    input.addEventListener('change', async () => {
      const file = input.files?.[0];
      input.value = '';
      if (!file) return;
      busy = true;
      uploadBtn.disabled = true;
      imageBtn.disabled = true;
      const meta = await handleUpload(file, {
        statusEl: status,
        kind,
        onDone: (m) => { selectedId = m.id; api.render(); onChange?.(api.getSelected()); },
      });
      busy = false;
      uploadBtn.disabled = false;
      imageBtn.disabled = false;
      if (meta) status.textContent = '';
    });
  };
  wireUpload(uploadBtn, fileInput, 'pdf');
  wireUpload(imageBtn, imageInput, 'image');

  pickers.add(api);
  // Load the list the first time a picker exists (no-op if signed out).
  listPdfs().then(() => api.render()).catch(() => api.render());
  api.render();
  return api;
}