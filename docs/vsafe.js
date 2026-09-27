/* Typesetting for the one symbol the interface repeats: every displayed
   "V_safe" becomes an italic V with a "safe" subscript. Static markup writes
   <i>V</i><sub>safe</sub> directly; scripts that build text from server data
   pass it through VSAFE.fmt. Attributes such as title and aria-label keep the
   plain "V_safe" text. Loaded before every other script. */
(() => {
  const TYPESET = '<i>V</i><sub>safe</sub>';
  // Markup in, markup out: only text outside tags is touched.
  window.VSAFE = {
    fmt: (html) => String(html ?? "").replace(/(<[^>]*>)|V_safe/g, (m, tag) => tag || TYPESET),
  };
})();
