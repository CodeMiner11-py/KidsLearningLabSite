/**
 * Loads shared header.html and footer.html fragments.
 * Optional data-root on the script tag (e.g. "../") for pages in subfolders.
 */
(function () {
  const script = document.currentScript;
  const root = (script && script.getAttribute("data-root")) || "";

  function loadPartial(url, targetId) {
    const el = document.getElementById(targetId);
    if (!el) return Promise.resolve();
    return fetch(root + url)
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.text();
      })
      .then(function (html) {
        el.innerHTML = html;
      })
      .catch(function (err) {
        console.error("KLL: failed to load " + url, err);
      });
  }

  // Loads a plain <script src> and resolves once it's actually executed —
  // used for kll-player.js / kll-iframe-modal.js so every page that
  // includes this file automatically gets the shared miniplayer and
  // iframe-modal helper without needing its own <script> tag.
  function loadScript(url) {
    return new Promise(function (resolve) {
      const el = document.createElement('script');
      el.src = root + url;
      el.onload = resolve;
      el.onerror = function () {
        console.error("KLL: failed to load " + url);
        resolve(); // don't block the rest of init() over one missing script
      };
      document.body.appendChild(el);
    });
  }

  function toggleMenu() {
    const container = document.getElementById("mobileNav");
    const hamburger = document.querySelector("#header .hamburger");
    if (container) container.classList.toggle("show");
    if (hamburger) hamburger.classList.toggle("open");
  }

  window.toggleMenu = toggleMenu;

  function init() {
    return Promise.all([
      loadPartial("header.html", "header"),
      loadPartial("footer.html", "footer"),
      loadScript("kll-player.js"),
      loadScript("kll-iframe-modal.js"),
    ]).then(function () {
      // Both header.html and footer.html are fetched/injected above, so
      // their buttons only exist in the DOM once this Promise.all
      // resolves — wiring their click handlers has to happen after, not
      // inside header.html/footer.html themselves (a <script> tag inside
      // fetched innerHTML never executes).
      wireHeaderFooterModals();
    });
  }

  // Account button (header) -> login iframe modal.
  // Newsletter button (footer) -> beehiiv iframe modal.
  // Shared by every page since header.html/footer.html are shared.
  function wireHeaderFooterModals() {
    const accountBtn = document.getElementById('kllAccountBtn');
    if (accountBtn) {
      accountBtn.addEventListener('click', function (e) {
        e.preventDefault();
        window.KLLIframeModal.open({
          title: 'Account',
          src: 'https://news.kidslearninglab.com/login',
          openInNewPageUrl: 'https://news.kidslearninglab.com/login',
        });
      });
    }

    const newsletterBtn = document.getElementById('kllNewsletterBtn');
    if (newsletterBtn) {
      newsletterBtn.addEventListener('click', function (e) {
        e.preventDefault();
        window.KLLIframeModal.open({
          title: 'Subscribe to the Newsletter',
          src: 'https://subscribe-forms.beehiiv.com/800fb944-0a45-4ab3-af09-fec64c932604',
          openInNewPageUrl: 'https://news.kidslearninglab.com',
        });
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

