/* Pipeline sidebar: preset choice, one-click Run, and a live chain of steps.
   Vanilla ES2020 to match index.html -- this project has no build step.
   The server is the authority on every timing: ticking numbers here are only
   for the step currently running, and they are replaced by the server's
   elapsed_ms the moment a step finishes. */
(() => {
  const $ = (id) => document.getElementById(id);
  const API = "/api/pipeline";
  const t = (en, ja) => window.I18N.t(en, ja);
  const esc = (text) => String(text ?? "").replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  // One element, both languages; i18n.js shows the one selected.
  const bilingual = (tag, cls, en, ja) =>
    `<${tag}${cls ? ` class="${cls}"` : ""} data-en="${esc(en)}" data-ja="${esc(ja ?? en)}">${window.I18N.fmt(esc(t(en, ja)))}</${tag}>`;
  const POST_HEADERS = { "Content-Type": "application/json", "X-5S-Pipeline": "1" };

  // Mirrors src/poi_params.py; the server validates every value again.
  const POI_TYPES = [
    ["school", "School", "学校"], ["hospital", "Hospital", "病院"], ["marketplace", "Market", "市場"],
    ["shop", "Shop", "商店"], ["bus_stop", "Bus stop", "バス停"],
  ];
  const defaultPoiParams = () => ({
    overture_min_confidence: 0.5,
    overture_top_percent: 95,
    speed_caps: Object.fromEntries(POI_TYPES.map(([t]) => [t, {
      enabled: t === "school", speed_kmh: 30, iso_min_urban: 3, iso_min_rural: 5,
    }])),
    sandwich_max_length_m: 250,
  });
  const loadPoiParams = () => {
    try {
      const saved = JSON.parse(localStorage.getItem("5s.poiParams") || "null");
      if (!saved) return defaultPoiParams();
      const base = defaultPoiParams();
      return {
        ...base, ...saved,
        speed_caps: Object.fromEntries(POI_TYPES.map(([t]) => [t, { ...base.speed_caps[t], ...(saved.speed_caps || {})[t] }])),
      };
    } catch (_) {
      return defaultPoiParams();
    }
  };

  // Mirrors src/aadt_estimation.py; the server validates every value again.
  // Only the scale factors are tunable: Scaling_to_2025 follows from the years
  // the probes were collected, and the clip bounds from what a road can carry.
  const AADT_REGIONS = [
    ["maharashtra", "Maharashtra", "マハラシュトラ", "aadt-scale-mh"],
    ["thailand", "Thailand", "タイ", "aadt-scale-th"],
  ];
  const defaultAadtParams = () => ({
    weighted_scale: { maharashtra: 1.0953284597075048, thailand: 0.045002453618601286 },
  });
  const loadAadtParams = () => {
    try {
      const saved = JSON.parse(localStorage.getItem("5s.aadtParams") || "null");
      const base = defaultAadtParams();
      if (!saved) return base;
      return { weighted_scale: { ...base.weighted_scale, ...(saved.weighted_scale || {}) } };
    } catch (_) {
      return defaultAadtParams();
    }
  };

  const state = {
    enabled: false,
    aadt: loadAadtParams(),
    poi: loadPoiParams(),
    preset: localStorage.getItem("5s.preset") || "quick",
    severity: localStorage.getItem("5s.severity") || "fatal",
    presets: {},
    runId: null,
    lastSeq: 0,
    running: false,
    clockOffset: 0, // server_ms - Date.now(), so a reload still shows true elapsed
    source: null,
    nodes: new Map(),
  };

  // ---- layout ------------------------------------------------------------
  const applyCollapsed = (collapsed) => {
    document.body.classList.toggle("sidebar-collapsed", collapsed);
    $("hamburger").setAttribute("aria-expanded", String(!collapsed));
    $("sidebar").setAttribute("aria-hidden", String(collapsed));
  };

  const trackResize = () => {
    // MapLibre only recomputes its canvas when told to; follow the whole
    // transition rather than snapping once it ends.
    const until = performance.now() + 260;
    const step = () => {
      if (window.map) window.map.resize();
      if (performance.now() < until) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };

  $("hamburger").addEventListener("click", () => {
    const collapsed = !document.body.classList.contains("sidebar-collapsed");
    applyCollapsed(collapsed);
    localStorage.setItem("5s.sidebar", collapsed ? "collapsed" : "open");
    trackResize();
  });

  if (window.ResizeObserver && $("map")) {
    new ResizeObserver(() => window.map && window.map.resize()).observe($("map"));
  }

  // ---- formatting --------------------------------------------------------
  const fmtElapsed = (ms) => {
    if (ms == null) return "";
    const total = Math.max(0, ms) / 1000;
    const m = Math.floor(total / 60);
    const s = (total - m * 60).toFixed(1).padStart(4, "0");
    return `${m}:${s}`;
  };

  // ---- chain rendering ---------------------------------------------------
  const GROUP_LABELS = {
    stage1: ["Stage 1 — whole segments", "ステージ1 — セグメント全体"],
    stage2: ["Stage 2 — after influence-zone split", "ステージ2 — 影響域分割後"],
  };

  const makeNode = (step) => {
    const li = document.createElement("li");
    li.className = step.group ? "node child" : "node";
    li.dataset.step = step.id;
    li.dataset.status = step.status || "pending";
    li.innerHTML =
      '<span class="rail"></span><span class="dot"></span>' +
      '<button class="head" type="button">' +
      bilingual("span", "label", step.label_en, step.label_ja) +
      '<span class="t"></span></button>' +
      '<div class="reason" hidden></div><pre class="log" hidden></pre>';
    const log = li.querySelector(".log");
    li.querySelector(".head").addEventListener("click", () => {
      log.hidden = !log.hidden;
    });
    return li;
  };

  const makeGroupHead = (group, count) => {
    const li = document.createElement("li");
    li.className = "node group-head";
    li.dataset.status = "pending";
    li.dataset.group = group;
    const [en, ja] = GROUP_LABELS[group] || [group, group];
    li.innerHTML =
      '<span class="rail"></span><span class="dot"></span>' +
      '<button class="head" type="button">' +
      bilingual("span", "label", en, ja) +
      `<span class="count">${count}</span></button>`;
    return li;
  };

  const renderChain = (steps) => {
    const chain = $("chain");
    chain.textContent = "";
    state.nodes.clear();
    const groupNodes = new Map();

    steps.forEach((step) => {
      if (step.group && !groupNodes.has(step.group)) {
        const count = steps.filter((s) => s.group === step.group).length;
        const head = makeGroupHead(step.group, count);
        chain.appendChild(head);
        groupNodes.set(step.group, { head, children: [], open: false });
        head.querySelector(".head").addEventListener("click", () => {
          const g = groupNodes.get(step.group);
          g.open = !g.open;
          g.children.forEach((c) => { c.hidden = !g.open; });
        });
      }
      const li = makeNode(step);
      if (step.group) {
        li.hidden = true;
        groupNodes.get(step.group).children.push(li);
      }
      chain.appendChild(li);
      state.nodes.set(step.id, li);
      if (step.status && step.status !== "pending") applyStep(step.id, step);
      else if (step.skip_reason) setReason(li, step.skip_reason, true);
      (step.log_tail || []).forEach((l) => appendLog(li, l));
    });

    state.groups = groupNodes;
  };

  const setReason = (li, reason, faded) => {
    const el = li.querySelector(".reason");
    if (!el) return;
    el.textContent = reason || "";
    el.hidden = !reason;
    if (faded) li.style.opacity = "0.65";
  };

  const appendLog = (li, entry) => {
    const pre = li.querySelector(".log");
    if (!pre) return;
    const span = document.createElement("span");
    if (entry.stream === "stderr") span.className = "err";
    span.textContent = entry.line + "\n";
    pre.appendChild(span);
    while (pre.childElementCount > 500) pre.removeChild(pre.firstChild);
    const pinned = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 24;
    if (pinned) pre.scrollTop = pre.scrollHeight;
  };

  const openGroupOf = (li) => {
    if (!state.groups) return;
    const group = li.classList.contains("child") ? li.dataset.group : null;
    state.groups.forEach((g, name) => {
      if (!g.children.includes(li)) return;
      g.head.dataset.status = "running";
      if (!g.open) { g.open = true; g.children.forEach((c) => { c.hidden = false; }); }
    });
    return group;
  };

  const applyStep = (stepId, data) => {
    const li = state.nodes.get(stepId);
    if (!li) return;
    const status = data.status;
    li.dataset.status = status;
    const t = li.querySelector(".t");

    if (status === "running") {
      li.startedMs = data.started_ms;
      li.style.opacity = "";
      setReason(li, null);
      li.querySelector(".log").hidden = false;
      openGroupOf(li);
      li.scrollIntoView({ block: "nearest" });
    } else {
      li.startedMs = null;
      if (t) t.textContent = status === "skipped" ? "—" : fmtElapsed(data.elapsed_ms);
      setReason(li, data.reason, status === "skipped");
      const log = li.querySelector(".log");
      if (log && status !== "failed" && status !== "timeout") log.hidden = true;
      markGroupsDone();
    }
  };

  const markGroupsDone = () => {
    if (!state.groups) return;
    state.groups.forEach((g) => {
      const states = g.children.map((c) => c.dataset.status);
      if (states.every((s) => s !== "pending" && s !== "running")) {
        g.head.dataset.status = states.some((s) => s === "failed" || s === "timeout")
          ? "failed"
          : states.every((s) => s === "skipped") ? "skipped" : "ok";
        if (g.open) { g.open = false; g.children.forEach((c) => { c.hidden = true; }); }
      }
    });
  };

  // one timer for every running node
  setInterval(() => {
    const now = Date.now() + state.clockOffset;
    state.nodes.forEach((li) => {
      if (li.dataset.status !== "running" || !li.startedMs) return;
      li.querySelector(".t").textContent = fmtElapsed(now - li.startedMs);
    });
  }, 100);

  // ---- summary -----------------------------------------------------------
  // Takes a function so the summary can be rebuilt in the other language.
  let summaryFn = null;
  const setSummary = (fn) => {
    summaryFn = fn;
    $("sb-summary").innerHTML = fn ? fn() : "";
  };

  // ---- presets -----------------------------------------------------------
  const ETA_JA = { "~10 s": "約 10 秒", minutes: "数分", hours: "数時間" };
  const renderPresets = () => {
    const box = $("presets");
    box.textContent = "";
    Object.entries(state.presets).forEach(([id, meta]) => {
      const label = document.createElement("label");
      label.className = "preset";
      label.innerHTML =
        `<input type="radio" name="preset" value="${id}"${id === state.preset ? " checked" : ""}>` +
        `<span class="preset-row">${bilingual("span", "preset-name", meta.label_en, meta.label_ja)}` +
        bilingual("span", "preset-eta", meta.eta, ETA_JA[meta.eta]) + "</span>" +
        bilingual("span", "preset-desc", meta.desc_en, meta.desc_ja) +
        `<span class="preset-avail" data-avail="${id}"></span>`;
      label.querySelector("input").addEventListener("change", () => selectPreset(id));
      box.appendChild(label);
    });
  };

  const selectPreset = async (id) => {
    state.preset = id;
    localStorage.setItem("5s.preset", id);
    const meta = state.presets[id] || {};
    const confirm = $("confirm");
    confirm.hidden = !meta.confirm;
    $("confirm-check").checked = false;
    if (meta.confirm) {
      const box = $("confirm-text");
      box.dataset.en = esc(meta.confirm_en);
      box.dataset.ja = esc(meta.confirm_ja);
      window.I18N.apply(confirm);
    }
    await loadSteps(id);
    updateRunButton();
  };

  const loadSteps = async (preset) => {
    const res = await fetch(`${API}/steps?preset=${preset}`);
    if (!res.ok) return;
    const data = await res.json();
    state.presets = data.presets || state.presets;
    if (!state.running) renderChain(data.steps);
    const avail = document.querySelector(`[data-avail="${preset}"]`);
    if (avail) {
      const n = data.steps.length - data.steps.filter((s) => s.skip_reason).length;
      avail.dataset.en = `${n} / ${data.steps.length} steps available`;
      avail.dataset.ja = `${data.steps.length} ステップ中 ${n} ステップを実行可能`;
      avail.textContent = t(avail.dataset.en, avail.dataset.ja);
    }
  };

  // ---- severity of the Elvik (2019) estimate -----------------------------
  // Chooses which coefficients the sens_exp_model step REPORTS; all three are
  // computed every run, so this never changes what the pipeline produces.
  const severityRadios = () => Array.from(document.querySelectorAll('input[name="severity"]'));

  const bindSeverity = () => {
    severityRadios().forEach((r) => {
      r.checked = r.value === state.severity;
      r.addEventListener("change", () => {
        state.severity = r.value;
        localStorage.setItem("5s.severity", r.value);
      });
    });
  };

  // ---- POI parameters (full / complete) --------------------------------------
  const savePoi = () => {
    try { localStorage.setItem("5s.poiParams", JSON.stringify(state.poi)); } catch (_) {}
  };
  const poiInputs = () => Array.from(document.querySelectorAll("#poi-opt input, #poi-reset"));

  const renderPoi = () => {
    $("poi-min-conf").value = state.poi.overture_min_confidence;
    $("poi-top-pct").value = state.poi.overture_top_percent;
    $("poi-sandwich-len").value = state.poi.sandwich_max_length_m;
    const body = $("poi-caps");
    body.textContent = "";
    POI_TYPES.forEach(([type, labelEn, labelJa]) => {
      const label = t(labelEn, labelJa);
      const cap = state.poi.speed_caps[type];
      const tr = document.createElement("tr");
      const name = document.createElement("td");
      name.textContent = label;
      const on = document.createElement("td");
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = cap.enabled;
      box.setAttribute("aria-label", t(`${label}: cap V_safe`, `${label}: V_safe に上限を適用`));
      box.addEventListener("change", () => { cap.enabled = box.checked; savePoi(); });
      on.appendChild(box);
      const speed = document.createElement("td");
      const num = document.createElement("input");
      num.type = "number"; num.min = "5"; num.max = "130"; num.step = "5";
      num.value = cap.speed_kmh;
      num.setAttribute("aria-label", t(`${label}: speed km/h`, `${label}: 速度 km/h`));
      num.addEventListener("change", () => { cap.speed_kmh = Number(num.value); savePoi(); });
      speed.appendChild(num);
      const minutes = [["iso_min_urban", t("urban", "市街地")], ["iso_min_rural", t("rural", "郊外")]].map(([key, where]) => {
        const td = document.createElement("td");
        const input = document.createElement("input");
        input.type = "number"; input.min = "1"; input.max = "30"; input.step = "1";
        input.value = cap[key];
        input.setAttribute("aria-label", t(`${label}: walking minutes, ${where}`, `${label}: 徒歩分数（${where}）`));
        input.addEventListener("change", () => { cap[key] = Number(input.value); savePoi(); });
        td.appendChild(input);
        return td;
      });
      tr.append(name, on, speed, ...minutes);
      body.appendChild(tr);
    });
    updateRunButton();
  };

  const bindPoi = () => {
    $("poi-min-conf").addEventListener("change", (e) => {
      state.poi.overture_min_confidence = Number(e.target.value); savePoi();
    });
    $("poi-top-pct").addEventListener("change", (e) => {
      state.poi.overture_top_percent = Number(e.target.value); savePoi();
    });
    $("poi-sandwich-len").addEventListener("change", (e) => {
      state.poi.sandwich_max_length_m = Number(e.target.value); savePoi();
    });
    $("poi-reset").addEventListener("click", () => { state.poi = defaultPoiParams(); savePoi(); renderPoi(); });
    renderPoi();
  };

  // ---- AADT scale factors (every preset) -------------------------------------
  const saveAadt = () => {
    try { localStorage.setItem("5s.aadtParams", JSON.stringify(state.aadt)); } catch (_) {}
  };
  const aadtInputs = () => Array.from(document.querySelectorAll("#aadt-opt input, #aadt-reset"));

  const renderAadt = () => {
    AADT_REGIONS.forEach(([region, labelEn, labelJa, id]) => {
      const input = $(id);
      input.value = state.aadt.weighted_scale[region];
      input.setAttribute("aria-label",
        t(`${labelEn}: vehicles per day per probe`, `${labelJa}: プローブ 1 台あたりの台数/日`));
    });
    updateRunButton();
  };

  const bindAadt = () => {
    AADT_REGIONS.forEach(([region, , , id]) => {
      $(id).addEventListener("change", (e) => {
        state.aadt.weighted_scale[region] = Number(e.target.value); saveAadt();
      });
    });
    $("aadt-reset").addEventListener("click", () => {
      state.aadt = defaultAadtParams(); saveAadt(); renderAadt();
    });
    renderAadt();
  };

  const updateRunButton = () => {
    const meta = state.presets[state.preset] || {};
    const needsConfirm = meta.confirm && !$("confirm-check").checked;
    $("btn-run").disabled = !state.enabled || state.running || needsConfirm;
    $("btn-cancel").hidden = !state.running;
    severityRadios().forEach((r) => { r.disabled = !state.enabled || state.running; });
    poiInputs().forEach((el) => { el.disabled = !state.enabled || state.running; });
    aadtInputs().forEach((el) => { el.disabled = !state.enabled || state.running; });
  };

  // ---- run lifecycle -----------------------------------------------------
  const startRun = async () => {
    $("btn-run").disabled = true;
    const res = await fetch(`${API}/run`, {
      method: "POST", headers: POST_HEADERS,
      body: JSON.stringify({ preset: state.preset, severity: state.severity,
                             poi_params: state.poi, aadt_params: state.aadt }),
    });
    const data = await res.json();
    if (!res.ok) {
      setSummary(() => `<span style="color:var(--c-pri)">${esc(data.error) || t("could not start", "開始できませんでした")}</span>`);
      updateRunButton();
      return;
    }
    state.runId = data.run_id;
    state.lastSeq = data.last_seq || 0;
    state.running = true;
    updateRunButton();
    setSummary(() => t("running…", "実行中…"));
    connect();
  };

  const cancelRun = async () => {
    $("btn-cancel").disabled = true;
    await fetch(`${API}/cancel`, {
      method: "POST", headers: POST_HEADERS,
      body: JSON.stringify({ run_id: state.runId }),
    }).catch(() => {});
    $("btn-cancel").disabled = false;
  };

  const connect = () => {
    if (state.source) state.source.close();
    const url = `${API}/events?run_id=${state.runId}&from_seq=${state.lastSeq}`;
    const source = new EventSource(url);
    state.source = source;

    const handle = (event) => {
      let data;
      try { data = JSON.parse(event.data); } catch { return; }
      if (data.seq <= state.lastSeq) return; // replay overlap after a reconnect
      state.lastSeq = data.seq;

      if (data.event === "step_start") applyStep(data.step_id, { status: "running", started_ms: data.started_ms });
      else if (data.event === "step_progress") {
        const li = state.nodes.get(data.step_id);
        if (li && li.dataset.status === "running") li.startedMs = Date.now() + state.clockOffset - data.elapsed_ms;
      } else if (data.event === "step_log") {
        const li = state.nodes.get(data.step_id);
        if (li) appendLog(li, data);
      } else if (data.event === "step_done") applyStep(data.step_id, data);
      else if (data.event === "run_done") finish(data);
    };

    ["step_start", "step_progress", "step_log", "step_done", "run_done"]
      .forEach((name) => source.addEventListener(name, handle));
    source.onerror = () => { /* EventSource reconnects on its own, resending Last-Event-ID */ };
  };

  const finish = (data) => {
    state.running = false;
    if (state.source) { state.source.close(); state.source = null; }
    updateRunButton();
    markGroupsDone();

    const c = data.counts || {};
    const tone = data.status === "ok" ? "var(--aurora)"
      : data.status === "partial" ? "var(--c-watch)" : "var(--c-pri)";
    const STATUS_JA = { ok: "成功", partial: "一部失敗", failed: "失敗", cancelled: "中止", timeout: "時間切れ" };
    const reloaded = !!(data.reload_pmtiles && window.reloadTiles);
    if (reloaded) window.reloadTiles(data.reload_pmtiles);
    setSummary(() => {
      let html = `<span style="color:${tone}">${esc(t(data.status, STATUS_JA[data.status]))}</span> · ` +
        `${fmtElapsed(data.elapsed_ms)} · ` +
        t(`${c.ok || 0} ok / ${c.skipped || 0} skipped / ${c.failed || 0} failed`,
          `成功 ${c.ok || 0} / スキップ ${c.skipped || 0} / 失敗 ${c.failed || 0}`);
      if (reloaded) {
        html += `<br><span style="color:var(--aurora)">${t("map reloaded from new tiles", "新しいタイルで地図を再読み込みしました")}</span>` +
          "<br>" + t("docs/segments_priority.pmtiles is tracked by git — commit it to publish.",
                     "docs/segments_priority.pmtiles は git で管理されています。公開するにはコミットしてください。");
      }
      return html;
    });
  };

  // ---- boot --------------------------------------------------------------
  const boot = async () => {
    applyCollapsed(localStorage.getItem("5s.sidebar") === "collapsed");

    let status;
    try {
      status = await (await fetch(`${API}/status`)).json();
    } catch {
      status = { enabled: false };
    }
    state.enabled = !!status.enabled;

    if (!state.enabled) {
      $("presets").innerHTML =
        `<p class="sb-note" data-en="Restart the server with &lt;code&gt;--enable-pipeline&lt;/code&gt; to run the pipeline from here."` +
        ` data-ja="ここからパイプラインを実行するには、&lt;code&gt;--enable-pipeline&lt;/code&gt; を付けてサーバーを再起動してください。"></p>`;
      window.I18N.apply($("presets"));
      updateRunButton();
      return;
    }

    await loadSteps(state.preset);
    renderPresets();
    await selectPreset(state.preset);

    const run = status.run;
    if (run) {
      state.runId = run.run_id;
      state.lastSeq = run.last_seq || 0;
      state.clockOffset = (run.server_ms || Date.now()) - Date.now();
      state.running = run.status === "running";
      renderChain(run.steps);
      if (state.running && !run.detached) {
        setSummary(() => t("running…", "実行中…"));
        connect();
      } else if (run.status === "interrupted") {
        setSummary(() => `<span style="color:var(--c-watch)">${t("interrupted — the server stopped mid-run",
          "中断されました（実行中にサーバーが停止しました）")}</span>`);
      } else if (run.status === "running" && run.detached) {
        setSummary(() => `<span style="color:var(--c-watch)">${t("a detached run is still alive; logs are unavailable",
          "切り離された実行がまだ動いています。ログは表示できません")}</span>`);
      } else {
        finish({ status: run.status, elapsed_ms: run.elapsed_ms, counts: run.counts });
      }
    }
    updateRunButton();
  };

  $("btn-run").addEventListener("click", startRun);
  $("btn-cancel").addEventListener("click", cancelRun);
  $("confirm-check").addEventListener("change", updateRunButton);
  bindSeverity();
  bindPoi();
  bindAadt();
  // i18n.js swaps every data-ja element itself; these are built from state.
  window.I18N.onChange(() => {
    renderPoi();
    renderAadt();
    if (summaryFn) setSummary(summaryFn);
  });
  boot();
})();
