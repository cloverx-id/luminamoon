(function () {
  var activeObserver = null;
  var activeScrollHandler = null;
  var activeProgressHandler = null;
  var activeResizeHandler = null;

  var STACK_MAP = {
    hf: ["loomy", "runtime"],
    fsdp: ["loomy", "runtime"],
    arrow: ["storage", "loomy"]
  };

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).catch(function () {
        return fallbackCopy(text);
      });
    }
    return Promise.resolve(fallbackCopy(text));
  }

  function fallbackCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand("copy");
    } catch (e) {
      /* ignore clipboard restrictions */
    }
    document.body.removeChild(ta);
  }

  function bindCopyButton(btn, getText) {
    if (!btn || btn.dataset.lmBound) return;
    btn.dataset.lmBound = "1";
    var baseLabel = btn.getAttribute("data-label") || btn.textContent.trim();
    var timer = null;
    btn.addEventListener("click", function () {
      var text = getText();
      if (!text) return;
      copyText(text).then(function () {
        if (timer) clearTimeout(timer);
        btn.textContent = "Copied";
        btn.classList.add("is-copied");
        timer = setTimeout(function () {
          btn.textContent = baseLabel;
          btn.classList.remove("is-copied");
          timer = null;
        }, 1500);
      });
    });
  }

  function bindPaletteTooltipGuard() {
    document.querySelectorAll('.md-header__option label[for^="__palette"]').forEach(function (lbl) {
      if (lbl.hasAttribute("title")) {
        if (!lbl.getAttribute("aria-label")) {
          lbl.setAttribute("aria-label", lbl.getAttribute("title"));
        }
        lbl.removeAttribute("title");
      }
    });
    document.querySelectorAll('input[name="__palette"]').forEach(function (input) {
      if (input.dataset.lmPaletteBound) return;
      input.dataset.lmPaletteBound = "1";
      input.addEventListener("change", function () {
        document.querySelectorAll(".md-tooltip--active").forEach(function (tip) {
          tip.classList.remove("md-tooltip--active");
        });
        if (document.activeElement && typeof document.activeElement.blur === "function") {
          document.activeElement.blur();
        }
      });
    });
  }

  function initReadingProgress(isHome) {
    if (activeProgressHandler) {
      window.removeEventListener("scroll", activeProgressHandler);
      activeProgressHandler = null;
    }
    var bar = document.querySelector(".lm-read-progress");
    if (isHome) {
      if (bar) bar.classList.remove("is-active");
      return;
    }
    if (!bar) {
      bar = document.createElement("div");
      bar.className = "lm-read-progress";
      bar.setAttribute("aria-hidden", "true");
      document.body.appendChild(bar);
    }

    function updateProgress() {
      var doc = document.documentElement;
      var maxScroll = doc.scrollHeight - window.innerHeight;
      if (maxScroll <= 120) {
        bar.classList.remove("is-active");
        bar.style.transform = "scaleX(0)";
        return;
      }
      var ratio = Math.min(1, Math.max(0, window.scrollY / maxScroll));
      bar.classList.toggle("is-active", window.scrollY > 24);
      bar.style.transform = "scaleX(" + ratio.toFixed(4) + ")";
    }

    activeProgressHandler = updateProgress;
    window.addEventListener("scroll", activeProgressHandler, { passive: true });
    updateProgress();
  }

  function teardown() {
    if (activeObserver) {
      activeObserver.disconnect();
      activeObserver = null;
    }
    if (activeScrollHandler) {
      window.removeEventListener("scroll", activeScrollHandler);
      activeScrollHandler = null;
    }
    if (activeResizeHandler) {
      window.removeEventListener("resize", activeResizeHandler);
      activeResizeHandler = null;
    }
  }

  function bindSearchShortcuts() {
    var searchInput = document.querySelector(".md-search__input");
    if (searchInput && searchInput.getAttribute("placeholder") === "Search") {
      searchInput.setAttribute("placeholder", "Search docs...");
    }
    if (window.__lmSearchShortcutBound) return;
    window.__lmSearchShortcutBound = true;
    document.addEventListener("keydown", function (e) {
      if ((e.ctrlKey || e.metaKey) && e.key && e.key.toLowerCase() === "k") {
        e.preventDefault();
        var toggle = document.querySelector('[data-md-toggle="search"]');
        var input = document.querySelector(".md-search__input");
        if (toggle && !toggle.checked) {
          toggle.checked = true;
          toggle.dispatchEvent(new Event("change", { bubbles: true }));
        }
        if (input) {
          input.click();
          input.focus();
          input.select();
        }
      }
    });
  }

  function initLmHome() {
    teardown();
    bindPaletteTooltipGuard();
    bindSearchShortcuts();

    var home = document.querySelector(".lm-home");
    initReadingProgress(!!home);
    if (!home) return;

    // 1. Install bar copy button
    home.querySelectorAll(".lm-copy-btn[data-copy]").forEach(function (btn) {
      bindCopyButton(btn, function () {
        return btn.getAttribute("data-copy") || "";
      });
    });

    // 2. Ecosystem code tabs (Sliding pill + Stack layer sync + ARIA + Copy)
    var tabList = home.querySelector(".lm-code-tabs");
    var indicator = home.querySelector(".lm-code-indicator");
    var tabs = Array.prototype.slice.call(home.querySelectorAll(".lm-code-tab"));
    var panels = Array.prototype.slice.call(home.querySelectorAll(".lm-code-panel"));
    var stackRows = Array.prototype.slice.call(home.querySelectorAll(".lm-stack-row[data-stack-layer]"));
    var codeCopyBtn = home.querySelector(".lm-code-copy-btn");

    function moveIndicator(activeTab) {
      if (!tabList || !indicator || !activeTab) return;
      indicator.style.width = activeTab.offsetWidth + "px";
      indicator.style.transform = "translateX(" + activeTab.offsetLeft + "px)";
      tabList.classList.add("has-indicator");
    }

    function syncStackLayers(key) {
      var activeLayers = STACK_MAP[key] || ["loomy"];
      stackRows.forEach(function (row) {
        var layer = row.getAttribute("data-stack-layer");
        row.classList.toggle("is-highlighted", activeLayers.indexOf(layer) !== -1);
      });
    }

    function activateTab(key, focusTab) {
      var currentTab = null;
      tabs.forEach(function (t) {
        var active = t.getAttribute("data-tab") === key;
        t.classList.toggle("is-active", active);
        t.setAttribute("aria-selected", active ? "true" : "false");
        t.setAttribute("tabindex", active ? "0" : "-1");
        if (active) {
          currentTab = t;
          if (focusTab) t.focus();
        }
      });
      panels.forEach(function (p) {
        var active = p.getAttribute("data-panel") === key;
        p.classList.toggle("is-active", active);
        p.hidden = !active;
      });
      moveIndicator(currentTab);
      syncStackLayers(key);
    }

    tabs.forEach(function (tab, idx) {
      if (tab.dataset.lmBound) return;
      tab.dataset.lmBound = "1";
      tab.addEventListener("click", function () {
        activateTab(tab.getAttribute("data-tab"), false);
      });
      tab.addEventListener("keydown", function (e) {
        if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
          e.preventDefault();
          var next = e.key === "ArrowRight"
            ? (idx + 1) % tabs.length
            : (idx - 1 + tabs.length) % tabs.length;
          activateTab(tabs[next].getAttribute("data-tab"), true);
        }
      });
    });

    var initialActive = home.querySelector(".lm-code-tab.is-active");
    if (initialActive) {
      requestAnimationFrame(function () {
        moveIndicator(initialActive);
      });
      syncStackLayers(initialActive.getAttribute("data-tab"));
    }

    activeResizeHandler = function () {
      var cur = home.querySelector(".lm-code-tab.is-active");
      if (cur) moveIndicator(cur);
    };
    window.addEventListener("resize", activeResizeHandler, { passive: true });

    bindCopyButton(codeCopyBtn, function () {
      var activePanel = home.querySelector(".lm-code-panel.is-active code");
      return activePanel ? activePanel.textContent : "";
    });

    // 3. Fast, zero-trap scroll arrival reveal
    var reveals = Array.prototype.slice.call(home.querySelectorAll(".lm-reveal"));
    if (reveals.length === 0) return;

    var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduceMotion || typeof IntersectionObserver === "undefined") {
      reveals.forEach(function (el) {
        el.classList.add("is-visible", "is-settled");
      });
      return;
    }

    function markVisible(el) {
      if (el.classList.contains("is-visible")) return;
      el.classList.add("is-visible");
      var delayMs = parseInt(el.style.getPropertyValue("--lm-delay") || "0", 10) || 0;
      setTimeout(function () {
        el.classList.add("is-settled");
      }, 620 + delayMs);
    }

    function sweepViewport() {
      var vh = window.innerHeight || document.documentElement.clientHeight;
      var remaining = 0;
      reveals.forEach(function (el) {
        if (el.classList.contains("is-visible")) return;
        var rect = el.getBoundingClientRect();
        if (rect.top < vh + 32) {
          markVisible(el);
        } else {
          remaining++;
        }
      });
      if (remaining === 0 && activeScrollHandler) {
        window.removeEventListener("scroll", activeScrollHandler);
        activeScrollHandler = null;
      }
    }

    document.documentElement.classList.add("lm-motion-ready");
    sweepViewport();

    activeObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting || entry.boundingClientRect.top < window.innerHeight) {
          markVisible(entry.target);
          activeObserver.unobserve(entry.target);
        }
      });
    }, { threshold: 0.04, rootMargin: "0px 0px -12px 0px" });

    reveals.forEach(function (el) {
      if (!el.classList.contains("is-visible")) activeObserver.observe(el);
    });

    activeScrollHandler = sweepViewport;
    window.addEventListener("scroll", activeScrollHandler, { passive: true });
  }

  if (window.__lmHomeInitialized) return;
  window.__lmHomeInitialized = true;

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(initLmHome);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initLmHome);
  } else {
    initLmHome();
  }
})();
