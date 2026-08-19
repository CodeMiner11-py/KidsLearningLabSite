/**
 * kll-iframe-modal.js — one generic "open an iframe in a modal" helper,
 * shared by the header's Account button, the footer's Newsletter button,
 * and contact.html's Contact Form button, so there's a single modal
 * implementation instead of three near-identical copies.
 */
(function (global) {
  'use strict';

  let overlay, titleEl, iframeEl, footerEl;

  function ensureMarkup() {
    if (overlay) return;
    overlay = document.createElement('div');
    overlay.className = 'kll-iframe-modal-overlay';
    overlay.id = 'kll-iframe-modal-overlay';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.innerHTML = `
      <div class="kll-iframe-modal">
        <div class="kll-iframe-modal-header">
          <div class="kll-iframe-modal-title" id="kll-iframe-modal-title"></div>
          <button type="button" class="kll-iframe-modal-close" id="kll-iframe-modal-close" aria-label="Close">✕</button>
        </div>
        <iframe id="kll-iframe-modal-frame" src="about:blank" title="Kids Learning Lab"></iframe>
        <div class="kll-iframe-modal-footer" id="kll-iframe-modal-footer"></div>
      </div>`;
    document.body.appendChild(overlay);

    titleEl  = document.getElementById('kll-iframe-modal-title');
    iframeEl = document.getElementById('kll-iframe-modal-frame');
    footerEl = document.getElementById('kll-iframe-modal-footer');

    document.getElementById('kll-iframe-modal-close').addEventListener('click', close);
    overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && overlay.classList.contains('open')) close(); });
  }

  // opts: { title, src, openInNewPageUrl (optional) }
  function open(opts) {
    ensureMarkup();
    titleEl.textContent = opts.title || '';
    iframeEl.src = opts.src;
    footerEl.innerHTML = opts.openInNewPageUrl
      ? `<a class="cta-button" href="${opts.openInNewPageUrl}" target="_blank" rel="noopener">Open in new page</a>`
      : '';
    overlay.classList.add('open');
    document.body.style.overflow = 'hidden';
  }

  function close() {
    if (!overlay) return;
    overlay.classList.remove('open');
    document.body.style.overflow = '';
    iframeEl.src = 'about:blank'; // stop whatever the iframe was doing (e.g. a form's in-progress state)
  }

  global.KLLIframeModal = { open, close };
})(window);
