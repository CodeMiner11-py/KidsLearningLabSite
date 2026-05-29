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
    ]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
