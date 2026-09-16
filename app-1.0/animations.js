// ============================================================
// animations.js — small shared library of reusable UI animations.
//
// Everything here is a plain function attached to `window.KLLAnim` so any
// module (learn.js, main.js, player.js, inline HTML handlers, etc.) can
// call it without an import — same pattern the app already uses for
// window.renderReviewPage. All animations are pure DOM/CSS-transition
// driven (no animation library), clean up after themselves (remove their
// own nodes when done), and no-op safely if their target element doesn't
// exist, so a missing DOM ref never throws mid-lesson.
// ============================================================

/**
 * Flies a floating "+N XP" label from the center of `container` to the
 * on-screen position of `target` (typically the running XP counter in
 * whichever corner it lives), shrinking and fading as it arrives.
 *
 * @param {Object} opts
 * @param {HTMLElement} opts.container - positioned ancestor the popup is
 *   absolutely placed within (e.g. #xpPopupLayer, which sits inset:0 over
 *   the lesson view). Falls back to document.body if omitted.
 * @param {HTMLElement} opts.target - element to shrink/fly toward (e.g.
 *   #lessonXpTracker). If omitted or not currently visible, the popup
 *   still plays its "appear, hold, fade in place" beats without a fly leg.
 * @param {number|string} opts.amount - XP amount, rendered as "+{amount} XP".
 * @param {number} [opts.holdMs=650] - how long the popup sits at full size
 *   in the center before flying/fading.
 * @param {Function} [opts.onDone] - called once the popup is fully done
 *   and its DOM node removed.
 */
function flyXpPopup({ container, target, amount, holdMs = 650, onDone } = {}) {
    const host = container || document.body;
    if (!host || amount == null) { onDone?.(); return; }
  
    const el = document.createElement('div');
    el.className = 'xp-popup';
    el.textContent = `+${amount} XP`;
    host.appendChild(el);
  
    // Force layout so the initial state is committed before we animate away
    // from it — otherwise the browser may coalesce the start/hold states.
    void el.offsetWidth;
  
    const flyAway = () => {
      if (target && target.isConnected && target.offsetParent !== null) {
        const hostRect = host.getBoundingClientRect();
        const targetRect = target.getBoundingClientRect();
        const startRect = el.getBoundingClientRect();
  
        const dx = (targetRect.left + targetRect.width / 2) - (startRect.left + startRect.width / 2);
        const dy = (targetRect.top + targetRect.height / 2) - (startRect.top + startRect.height / 2);
  
        el.style.transition = 'transform .45s cubic-bezier(.4,0,.2,1), opacity .45s ease';
        el.style.transform = `translate(-50%, -50%) translate(${dx}px, ${dy}px) scale(0.32)`;
        el.style.opacity = '0';
  
        // Bump the corner counter's own tiny "pulse" right as the popup lands.
        window.setTimeout(() => pulse(target), 380);
      } else {
        // No valid fly target — just fade out in place.
        el.style.transition = 'transform .35s ease, opacity .35s ease';
        el.style.transform = 'translate(-50%, -50%) scale(0.9)';
        el.style.opacity = '0';
      }
  
      window.setTimeout(() => {
        el.remove();
        onDone?.();
      }, 480);
    };
  
    // Small pop-in bounce, then hold, then fly.
    el.style.transform = 'translate(-50%, -50%) scale(0.4)';
    el.style.opacity = '0';
    el.style.transition = 'transform .22s cubic-bezier(.34,1.56,.64,1), opacity .18s ease';
    requestAnimationFrame(() => {
      el.style.transform = 'translate(-50%, -50%) scale(1)';
      el.style.opacity = '1';
      window.setTimeout(flyAway, holdMs);
    });
  }
  
  /**
   * Quick scale-pulse on any element — used to make the destination XP
   * counter visibly "receive" the popup instead of just changing its number.
   */
  function pulse(el, { scale = 1.18, ms = 220 } = {}) {
    if (!el) return;
    const prevTransition = el.style.transition;
    el.style.transition = `transform ${ms}ms cubic-bezier(.34,1.56,.64,1)`;
    el.style.transform = `scale(${scale})`;
    window.setTimeout(() => {
      el.style.transform = 'scale(1)';
      window.setTimeout(() => { el.style.transition = prevTransition; }, ms);
    }, ms);
  }
  
  /**
   * Momentary "press" feedback for any tappable element — a quick scale-down
   * and back, for buttons/cards that don't already have their own affordance.
   * Safe to call on every pointerdown; it's cheap and self-cleans.
   */
  function tapBounce(el) {
    if (!el) return;
    el.style.transition = 'transform .12s ease';
    el.style.transform = 'scale(0.95)';
    window.setTimeout(() => {
      el.style.transform = 'scale(1)';
    }, 120);
  }
  
  /**
   * A short burst of small colored dots from a point (e.g. a completed
   * checkbox, a claimed reward) — a lightweight confetti-lite effect that
   * doesn't require a canvas or external library.
   *
   * @param {HTMLElement} originEl - element to burst outward from.
   * @param {Object} [opts]
   * @param {string[]} [opts.colors] - dot colors to cycle through.
   * @param {number} [opts.count=10] - number of dots.
   */
  function confettiBurst(originEl, { colors, count = 10 } = {}) {
    if (!originEl) return;
    const palette = colors || ['#FF8A2B', '#2FA84F', '#1E6FE0', '#7A5CFF', '#FFC93C'];
    const rect = originEl.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
  
    const layer = document.createElement('div');
    layer.style.cssText = 'position:fixed; inset:0; pointer-events:none; z-index:9999;';
    document.body.appendChild(layer);
  
    for (let i = 0; i < count; i++) {
      const dot = document.createElement('div');
      const angle = (Math.PI * 2 * i) / count + Math.random() * 0.4;
      const dist = 40 + Math.random() * 30;
      const size = 5 + Math.random() * 4;
      dot.style.cssText = `
        position:absolute; left:${cx}px; top:${cy}px;
        width:${size}px; height:${size}px; border-radius:50%;
        background:${palette[i % palette.length]};
        transform:translate(-50%,-50%) scale(1); opacity:1;
        transition: transform .55s cubic-bezier(.22,.9,.32,1), opacity .55s ease;
      `;
      layer.appendChild(dot);
      requestAnimationFrame(() => {
        const dx = Math.cos(angle) * dist;
        const dy = Math.sin(angle) * dist;
        dot.style.transform = `translate(${dx - size / 2}px, ${dy - size / 2}px) scale(0.3)`;
        dot.style.opacity = '0';
      });
    }
  
    window.setTimeout(() => layer.remove(), 650);
  }
  
  /**
   * Simple fade+rise pop-in for any element that just entered the DOM (a
   * newly rendered card, a toast, a modal's inner content) — call right
   * after setting its innerHTML/appendChild.
   */
  function popIn(el, { distance = 8, ms = 240 } = {}) {
    if (!el) return;
    el.style.opacity = '0';
    el.style.transform = `translateY(${distance}px)`;
    el.style.transition = `opacity ${ms}ms ease, transform ${ms}ms cubic-bezier(.22,.9,.32,1)`;
    requestAnimationFrame(() => {
      el.style.opacity = '1';
      el.style.transform = 'translateY(0)';
    });
  }
  
  window.KLLAnim = { flyXpPopup, pulse, tapBounce, confettiBurst, popIn };