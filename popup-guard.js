(function () {
    var ALLOWED_KEYWORD = "spotify";
    var currentHost = window.location.hostname;
  
    function isSameSiteOrSubdomain(hostname) {
      return hostname === currentHost || hostname.endsWith("." + currentHost) || currentHost.endsWith("." + hostname);
    }
  
    function isAllowed(url) {
      try {
        var u = new URL(url, window.location.href);
        if (u.protocol !== "http:" && u.protocol !== "https:") return true;
        if (isSameSiteOrSubdomain(u.hostname)) return true;
        if (u.hostname.toLowerCase().indexOf(ALLOWED_KEYWORD) !== -1) return true;
        return false;
      } catch (e) {
        return true;
      }
    }
  
    function showPopup(url, onLeave) {
      var overlay = document.createElement("div");
      overlay.style.cssText =
        "position:fixed;top:0;left:0;width:100%;height:100%;" +
        "background:rgba(0,0,0,0.5);z-index:999999;display:flex;" +
        "align-items:center;justify-content:center;font-family:Arial,sans-serif;";
  
      var box = document.createElement("div");
      box.style.cssText =
        "background:#fff;border-radius:12px;padding:24px;max-width:360px;" +
        "width:90%;text-align:center;box-shadow:0 10px 30px rgba(0,0,0,0.3);";
  
      var img = document.createElement("img");
      img.src = "https://kidslearninglab.com/wp-content/uploads/2025/02/podcast-logo-app-rounded.png";
      img.alt = "Logo";
      img.style.cssText = "width:64px;height:64px;border-radius:16px;margin-bottom:12px;";
  
      var title = document.createElement("h2");
      title.textContent = "Are You Sure?";
      title.style.cssText = "margin:0 0 8px 0;font-size:20px;color:#111;";
  
      var msg = document.createElement("p");
      msg.textContent = "Do you want to leave the site to go to " + url + "?";
      msg.style.cssText = "margin:0 0 20px 0;font-size:14px;color:#444;word-break:break-all;";
  
      var btnRow = document.createElement("div");
      btnRow.style.cssText = "display:flex;flex-direction:column;gap:10px;";
  
      var leaveBtn = document.createElement("button");
      leaveBtn.textContent = "Leave " + currentHost;
      leaveBtn.style.cssText =
        "background:#e53935;color:#fff;border:none;padding:12px;border-radius:8px;" +
        "font-size:15px;cursor:pointer;";
      leaveBtn.onclick = function () {
        document.body.removeChild(overlay);
        onLeave();
      };
  
      var stayBtn = document.createElement("button");
      stayBtn.textContent = "Not now";
      stayBtn.style.cssText =
        "background:#eee;color:#333;border:none;padding:12px;border-radius:8px;" +
        "font-size:15px;cursor:pointer;";
      stayBtn.onclick = function () {
        document.body.removeChild(overlay);
      };
  
      btnRow.appendChild(leaveBtn);
      btnRow.appendChild(stayBtn);
  
      box.appendChild(img);
      box.appendChild(title);
      box.appendChild(msg);
      box.appendChild(btnRow);
      overlay.appendChild(box);
      document.body.appendChild(overlay);
    }
  
    document.addEventListener("click", function (e) {
      var link = e.target.closest && e.target.closest("a[href]");
      if (!link) return;
  
      var href = link.getAttribute("href");
      if (!href || href.startsWith("#") || href.startsWith("javascript:")) return;
  
      if (!isAllowed(href)) {
        e.preventDefault();
        var target = link.target;
        var absoluteUrl = link.href;
        showPopup(absoluteUrl, function () {
          if (target === "_blank") {
            window.open(absoluteUrl, "_blank");
          } else {
            window.location.href = absoluteUrl;
          }
        });
      }
    }, true);
  })();