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

  // opts: {
  //   title,
  //   src,
  //   openInNewPageUrl (optional) — renders an "Open in new page" button,
  //   footerLinks (optional) — [{ label, src }]. Each renders as a text link
  //     below the button; clicking it loads `src` inside this modal's iframe.
  // }
  function open(opts) {
    ensureMarkup();
    titleEl.textContent = opts.title || '';
    iframeEl.src = opts.src;

    let html = '';
    if (opts.openInNewPageUrl) {
      html += `<a class="cta-button" href="${opts.openInNewPageUrl}" target="_blank" rel="noopener">Open in new page</a>`;
    }
    if (opts.footerLinks && opts.footerLinks.length) {
      html += '<div class="kll-iframe-modal-links" style="display:flex;flex-direction:column;align-items:center;gap:8px;margin-top:12px;">';
      opts.footerLinks.forEach((l, i) => {
        html += `<a href="${l.src}" data-link-idx="${i}" style="font-size:14px;color:inherit;text-decoration:underline;cursor:pointer;">${l.label}</a>`;
      });
      html += '</div>';
    }
    footerEl.innerHTML = html;

    if (opts.footerLinks && opts.footerLinks.length) {
      footerEl.querySelectorAll('[data-link-idx]').forEach((a) => {
        a.addEventListener('click', (e) => {
          e.preventDefault();
          iframeEl.src = opts.footerLinks[Number(a.dataset.linkIdx)].src;
        });
      });
    }

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
