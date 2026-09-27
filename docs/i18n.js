/* One display language at a time: English (default) or Japanese.
   An element with a data-ja attribute shows its English markup (kept in
   data-en, filled from the element on first use) or its Japanese one; the
   same goes for data-ja-title, data-ja-aria-label and data-ja-href on those attributes.
   Scripts that build text call I18N.t(en, ja), and re-render through
   I18N.onChange when the language switches. Loaded before every other script.
   Every displayed "V_safe" is typeset as an italic V with "safe" subscript
   (I18N.fmt); attributes such as title and aria-label keep the plain text. */
(() => {
  const KEY = "5s.lang";
  const LANGS = ["en", "ja"];
  const listeners = [];
  let lang = "en";
  // ?lang=ja in the URL wins over the remembered choice, so a link can open in either language.
  const fromUrl = new URLSearchParams(location.search).get("lang");
  try {
    const saved = localStorage.getItem(KEY);
    if (LANGS.includes(saved)) lang = saved;
  } catch (_) {}
  if (LANGS.includes(fromUrl)) lang = fromUrl;

  const ATTRS = [["title", "jaTitle", "enTitle"], ["aria-label", "jaAriaLabel", "enAriaLabel"],
                 ["href", "jaHref", "enHref"]];

  // Markup in, markup out: only text outside tags is touched.
  const VSAFE = '<span style="font-style: italic;">V</span><sub>safe</sub>';
  const fmt = (html) => String(html ?? "").replace(/(<[^>]*>)|V_safe/g, (m, tag) => tag || VSAFE);

  const apply = (root = document) => {
    document.documentElement.lang = lang;
    root.querySelectorAll("[data-ja]").forEach((el) => {
      if (el.dataset.en === undefined) el.dataset.en = el.innerHTML;
      el.innerHTML = fmt(lang === "ja" ? el.dataset.ja : el.dataset.en);
    });
    ATTRS.forEach(([attr, jaKey, enKey]) => {
      root.querySelectorAll(`[data-ja-${attr}]`).forEach((el) => {
        if (el.dataset[enKey] === undefined) el.dataset[enKey] = el.getAttribute(attr) || "";
        el.setAttribute(attr, lang === "ja" ? el.dataset[jaKey] : el.dataset[enKey]);
      });
    });
    document.querySelectorAll("[data-lang-choice]").forEach((b) => {
      b.setAttribute("aria-pressed", String(b.dataset.langChoice === lang));
    });
  };

  window.I18N = {
    get lang() { return lang; },
    t: (en, ja) => (lang === "ja" && ja != null ? ja : en),
    fmt,
    apply,
    onChange: (fn) => listeners.push(fn),
    set: (next) => {
      if (!LANGS.includes(next) || next === lang) return;
      lang = next;
      try { localStorage.setItem(KEY, lang); } catch (_) {}
      apply();
      listeners.forEach((fn) => fn(lang));
    },
  };

  const boot = () => {
    document.querySelectorAll("[data-lang-choice]").forEach((b) => {
      b.addEventListener("click", () => window.I18N.set(b.dataset.langChoice));
    });
    apply();
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
