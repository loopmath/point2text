(() => {
  'use strict';

  const $ = id => document.getElementById(id);
  const NS = 'http://www.w3.org/2000/svg';
  const colors = ['#7b968a', '#8c91ad', '#b69a76', '#8f9f66', '#b58b8a', '#7196a4', '#a38dad', '#a5a179', '#557b78', '#a0768d'];
  const plot = { left: 54, top: 22, width: 718, height: 490 };
  let previousSeed = null;

  function freshSeed() {
    const limit = 2 ** 31 - 7;
    const random = new Uint32Array(1);
    crypto.getRandomValues(random);
    let seed = random[0] % (limit + 1);
    if (seed === previousSeed) seed = (seed + 1) % (limit + 1);
    previousSeed = seed;
    return seed;
  }
  const state = { map: null, status: null, target: null, runId: null, runTrace: null, busy: false, epoch: 0, candidates: [], selected: -1, tab: 'best', started: 0, runElapsed: 0, requested: 6, topicColors: new Map(), transform: null, baseScale: null, runTimer: null, statusTimer: null, tickTimer: null, mapLoading: false, hovered: null, saved: [], saving: false, panMode: false, pointers: new Map(), gesture: null, suppressClick: false, draft: null, draftSequence: 0, draftTokens: [] };

  function element(tag, attrs = {}, text) {
    const el = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
    if (text !== undefined) el.textContent = text;
    return el;
  }

  function svgElement(tag, attrs = {}, text) {
    const el = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
    if (text !== undefined) el.textContent = text;
    return el;
  }

  function appendSVG(parent, tag, attrs, text) {
    const el = svgElement(tag, attrs, text);
    parent.append(el);
    return el;
  }

  function announce(message) { $('announcement').textContent = message; }
  function finite(value) { return typeof value === 'number' && Number.isFinite(value); }
  function format(value, digits = 3) { return finite(value) ? value.toFixed(digits).replace(/^-0\.0+$/, (0).toFixed(digits)) : '—'; }
  function coordinateText(point) { const digits = state.status?.generator?.refinement ? 6 : 3; return `${format(point[0], digits)}, ${format(point[1], digits)}`; }
  function distanceText(value) { return finite(value) ? format(value, value < 0.01 ? 6 : 3) : '—'; }
  function topicColor(topic) { return state.topicColors.get(topic) || colors[0]; }

  async function api(path, options = {}) {
    const response = await fetch(path, { ...options, headers: { 'Content-Type': 'application/json', ...options.headers } });
    let data;
    try { data = await response.json(); } catch { throw new Error(`The local worker returned an unreadable response (${response.status}).`); }
    if (!response.ok) throw new Error(typeof data.error === 'string' ? data.error : typeof data.detail === 'string' ? data.detail : `The local worker returned ${response.status}.`);
    return data;
  }

  function setConnection(text, kind) {
    $('connection-text').textContent = text;
    $('connection').className = `connection ${kind || ''}`;
  }

  function updateControls() {
    const available = Boolean(state.status?.generator?.available);
    $('generate').disabled = !state.map || !state.target || !available || state.busy;
    $('generate').hidden = state.busy;
    $('cancel').hidden = !state.busy;
    $('candidate-count').disabled = state.busy;
    $('guidance-control').hidden = state.status?.generator?.guided !== true;
    $('guidance-updates').disabled = state.busy;
    $('refinement-control').hidden = state.status?.generator?.refinement !== true;
    $('refinement-epsilon').disabled = state.busy;
    $('refinement-seconds').disabled = state.busy || $('refinement-unlimited').checked;
    $('refinement-unlimited').disabled = state.busy;
    $('refinement-restarts').disabled = state.busy;
    $('refinement-restart-candidates').disabled = state.busy || Number($('refinement-restarts').value) === 0;
    $('candidate-label').textContent = state.status?.generator?.refinement ? 'INITIAL CANDIDATES' : 'CANDIDATES';
    $('clear-target').hidden = !state.target;
    updateSaveControl();
    renderDraftPanel();
    for (const id of ['zoom-in', 'zoom-out', 'fit-map', 'pan-mode']) $(id).disabled = !state.map;
    if (state.busy) {
      $('action-help').textContent = 'Each completed paragraph is scored at its actual map position.';
    } else if (!state.map) {
      $('action-help').textContent = 'The local encoder is preparing the corpus.';
    } else if (!available) {
      $('action-help').textContent = 'The local generator needs attention. See Map & local model details below.';
    } else if (!state.target) {
      $('action-help').textContent = 'Select a target to begin.';
    } else if (state.candidates.length) {
      $('action-help').textContent = 'Choose another position, or search again for a closer paragraph.';
    } else {
      $('action-help').textContent = 'Paragraphs will be ranked by their measured distance to your target.';
    }
  }

  async function checkStatus() {
    try {
      const status = await api('/api/status');
      state.status = status;
      const diffusion = Boolean(status.generator?.runtime);
      const refinement = status.generator?.refinement === true;
      document.querySelector('.mode-label').textContent = refinement ? 'Canonical refinement · experimental' : diffusion ? status.generator.guided ? 'Experimental diffusion guidance' : 'Unguided diffusion' : 'Local candidate search';
      document.querySelector('.method-note p').textContent = refinement ? 'Diffusion proposes text and span repairs; canonical encoder gradients propose local edits. We keep measured improvements. Target tolerance is confirmed with fresh text scores; text quality remains unverified.' : diffusion && status.generator.guided ? 'A frozen diffusion model refines token choices with an approximate map-gradient signal. We re-embed the finished text and keep the closest measured paragraph.' : 'We generate candidates, embed their actual text, and keep the closest match. The line shows the remaining distance in this two-dimensional map.';
      $('detail-generator').textContent = status.generator?.model || 'No model selected';
      const problems = [status.error, status.generator?.error].filter(Boolean);
      $('detail-status').textContent = problems.length ? problems.join('\n') : status.ready ? status.generator?.available ? 'Local worker ready. Models remain frozen.' : 'Local generator unavailable. Start Ollama and load the model listed above. This page checks again automatically.' : 'Encoding the corpus and preparing its fixed projection.';
      if (status.ready) {
        setConnection(status.generator?.available ? 'Local · ready' : 'Generator unavailable', status.generator?.available ? 'ready' : 'error');
        if (!state.map && !state.mapLoading) await loadMap();
      } else if (status.error) {
        setConnection('Needs attention', 'error');
        showMapError('The map could not be prepared', status.error);
      } else {
        setConnection('Preparing map', '');
      }
      updateControls();
    } catch (error) {
      setConnection('Worker unavailable', 'error');
      $('detail-status').textContent = error.message;
      if (!state.map) showMapError('Waiting for the local worker', 'Keep the local app running. This page will reconnect automatically.');
    } finally {
      state.statusTimer = window.setTimeout(checkStatus, state.map && state.status?.generator?.available ? 15000 : 1500);
    }
  }

  function showMapError(title, message) {
    $('map-overlay').hidden = false;
    $('loading-title').textContent = title;
    $('loading-copy').textContent = message;
  }

  async function loadMap() {
    state.mapLoading = true;
    try {
      const data = await api('/api/map');
      if (!Array.isArray(data.points) || !data.bounds || !data.map_id) throw new Error('The map response is incomplete.');
      const points = data.points.filter(p => finite(p.x) && finite(p.y));
      if (!points.length) throw new Error('The corpus has no plotted paragraphs.');
      prepareTransform(data.bounds);
      state.map = { ...data, points };
      const topics = data.topics?.length ? data.topics : [...new Set(points.map(p => p.topic))];
      topics.forEach((topic, i) => state.topicColors.set(topic, colors[i % colors.length]));
      renderMap();
      renderLegend(topics);
      $('map-overlay').hidden = true;
      $('plot-instruction').hidden = false;
      $('map-meta').textContent = `${data.corpus_size || points.length} paragraphs · ${topics.length} topics`;
      $('detail-encoder').textContent = `${data.encoder?.id || 'Local encoder'} · ${data.encoder?.dimensions || '—'} dimensions`;
      if (data.explained_variance?.length >= 2) {
        const [pc1, pc2] = data.explained_variance;
        $('detail-projection').textContent = `Fixed PCA · PC1 ${(pc1 * 100).toFixed(1)}% · PC2 ${(pc2 * 100).toFixed(1)}% of variance. Equal scale on both axes.`;
      }
      announce(`Map ready. ${points.length} corpus paragraphs. Click to choose a target.`);
      await loadSaved();
    } catch (error) {
      showMapError('The map is not ready yet', error.message);
    } finally {
      state.mapLoading = false;
      updateControls();
    }
  }

  function prepareTransform(bounds) {
    const xmin = Number(bounds.xmin), xmax = Number(bounds.xmax), ymin = Number(bounds.ymin), ymax = Number(bounds.ymax);
    if (![xmin, xmax, ymin, ymax].every(Number.isFinite)) throw new Error('The map bounds are invalid.');
    const dx = Math.max(xmax - xmin, .001), dy = Math.max(ymax - ymin, .001);
    const scale = Math.min(plot.width / (dx * 1.18), plot.height / (dy * 1.18));
    const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2;
    if (!state.baseScale) state.baseScale = scale;
    setView(scale, cx, cy);
  }

  function setView(scale, cx, cy) {
    const sx = plot.left + plot.width / 2, sy = plot.top + plot.height / 2;
    state.transform = {
      scale, cx, cy, sx, sy,
      toScreen: (x, y) => [sx + (x - cx) * scale, sy - (y - cy) * scale],
      toWorld: (x, y) => [cx + (x - sx) / scale, cy - (y - sy) / scale],
      minX: cx - plot.width / (2 * scale), maxX: cx + plot.width / (2 * scale),
      minY: cy - plot.height / (2 * scale), maxY: cy + plot.height / (2 * scale)
    };
    $('zoom-level').textContent = `${Math.round(scale / state.baseScale * 100)}%`;
  }

  function zoomAt(factor, point = [plot.left + plot.width / 2, plot.top + plot.height / 2]) {
    if (!state.transform) return;
    const t = state.transform, anchor = t.toWorld(...point);
    const scale = Math.max(state.baseScale / 8, Math.min(state.baseScale * 100, t.scale * factor));
    setView(scale, anchor[0] - (point[0] - t.sx) / scale, anchor[1] + (point[1] - t.sy) / scale);
    hideTooltip(); renderMap();
  }

  function niceStep(span) {
    const raw = span / 5;
    const magnitude = 10 ** Math.floor(Math.log10(raw));
    const fraction = raw / magnitude;
    return (fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10) * magnitude;
  }

  function frameMap() {
    if (!state.map) return;
    const bounds = { ...state.map.bounds };
    const include = (x, y) => {
      bounds.xmin = Math.min(bounds.xmin, x); bounds.xmax = Math.max(bounds.xmax, x);
      bounds.ymin = Math.min(bounds.ymin, y); bounds.ymax = Math.max(bounds.ymax, y);
    };
    if (state.target) include(...state.target);
    state.candidates.forEach(candidate => include(candidate.x, candidate.y));
    if ($('show-saved').checked) state.saved.forEach(item => include(item.x, item.y));
    prepareTransform(bounds);
    renderMap();
  }

  function tickLabel(value, step) {
    const digits = Math.min(5, Math.max(0, -Math.floor(Math.log10(step))));
    return format(value, digits);
  }

  function renderMap() {
    const t = state.transform;
    const grid = $('grid-layer'); grid.replaceChildren();
    const xStep = niceStep(t.maxX - t.minX), yStep = niceStep(t.maxY - t.minY);
    for (let x = Math.ceil(t.minX / xStep) * xStep; x <= t.maxX + xStep * .001; x += xStep) {
      const [sx] = t.toScreen(x, 0);
      appendSVG(grid, 'line', { x1: sx, y1: plot.top, x2: sx, y2: plot.top + plot.height, class: Math.abs(x) < xStep * .001 ? 'axis-line' : 'grid-line' });
      appendSVG(grid, 'text', { x: sx, y: plot.top + plot.height + 19, 'text-anchor': 'middle', class: 'axis-tick' }, tickLabel(x, xStep));
    }
    for (let y = Math.ceil(t.minY / yStep) * yStep; y <= t.maxY + yStep * .001; y += yStep) {
      const [, sy] = t.toScreen(0, y);
      appendSVG(grid, 'line', { x1: plot.left, y1: sy, x2: plot.left + plot.width, y2: sy, class: Math.abs(y) < yStep * .001 ? 'axis-line' : 'grid-line' });
      appendSVG(grid, 'text', { x: plot.left - 10, y: sy + 3.5, 'text-anchor': 'end', class: 'axis-tick' }, tickLabel(y, yStep));
    }
    const layer = $('points-layer'); layer.replaceChildren();
    state.map.points.forEach(point => {
      const [x, y] = t.toScreen(point.x, point.y);
      const dot = appendSVG(layer, 'circle', { cx: x, cy: y, r: 4, fill: topicColor(point.topic), class: 'corpus-point' });
      appendSVG(dot, 'title', {}, `${point.title} · ${point.topic}`);
      dot.addEventListener('pointerenter', () => showTooltip(point, dot));
      dot.addEventListener('pointerleave', hideTooltip);
      dot.addEventListener('click', event => {
        event.stopPropagation();
        setTarget([point.x, point.y]);
        showSource(point);
      });
    });
    renderSavedPoints(); renderTarget(); renderResults();
  }

  function renderLegend(topics) {
    $('legend').replaceChildren();
    topics.forEach(topic => {
      const item = element('span', { class: 'legend-item' });
      const dot = element('i'); dot.style.backgroundColor = topicColor(topic);
      item.append(dot, element('span', {}, topic));
      $('legend').append(item);
    });
  }

  function showTooltip(point, dot) {
    state.hovered = dot;
    dot.classList.add('hovered'); dot.setAttribute('r', '6');
    $('tooltip-topic').textContent = point.topic || 'Corpus paragraph';
    $('tooltip-title').textContent = point.title || 'Untitled';
    const [x, y] = state.transform.toScreen(point.x, point.y);
    const shell = $('plot-shell');
    const px = x / 800 * shell.clientWidth;
    const py = y / 565 * $('map').getBoundingClientRect().height;
    $('point-tooltip').style.left = `${Math.max(8, Math.min(px + 12, shell.clientWidth - 242))}px`;
    $('point-tooltip').style.top = `${Math.max(8, py - 72)}px`;
    $('point-tooltip').hidden = false;
  }

  function hideTooltip() {
    if (state.hovered) { state.hovered.classList.remove('hovered'); state.hovered.setAttribute('r', '4'); }
    state.hovered = null;
    $('point-tooltip').hidden = true;
  }

  function showSource(point) {
    $('source-topic').textContent = point.topic || 'Corpus paragraph';
    $('source-title').textContent = point.title || 'Untitled';
    $('source-text').textContent = point.text || '';
    $('source-preview').hidden = false;
  }

  function renderNeighbors() {
    $('neighbors').replaceChildren();
    if (!state.target) {
      $('neighbor-caption').textContent = 'Select a point to explore its neighborhood';
      $('neighbors').append(element('p', { class: 'empty-context' }, 'Every dot is a real paragraph. Hover to inspect it, or select a position between ideas.'));
      return;
    }
    const [x, y] = state.target;
    const closest = state.map.points.map(p => ({ point: p, distance: Math.hypot(p.x - x, p.y - y) })).sort((a, b) => a.distance - b.distance).slice(0, 4);
    $('neighbor-caption').textContent = 'Closest corpus paragraphs';
    closest.forEach(({ point, distance }) => {
      const button = element('button', { type: 'button', class: 'neighbor', title: `Read ${point.title}` });
      const dot = element('span', { class: 'neighbor-dot' }); dot.style.backgroundColor = topicColor(point.topic);
      const label = element('span'); label.append(element('strong', {}, point.title || 'Untitled'), element('small', {}, `${point.topic || 'Corpus'} · ${format(distance)} away`));
      button.append(dot, label); button.addEventListener('click', () => showSource(point));
      $('neighbors').append(button);
    });
  }

  async function quietlyCancel(id) {
    if (!id) return;
    try { await api(`/api/runs/${encodeURIComponent(id)}`, { method: 'DELETE' }); } catch { /* A stale run must not replace the current UI. */ }
  }

  function abandonRun() {
    const oldId = state.runId;
    state.epoch += 1;
    state.runId = null; state.busy = false;
    window.clearTimeout(state.runTimer); window.clearInterval(state.tickTimer);
    if (oldId) void quietlyCancel(oldId);
  }

  function resetOutput() {
    state.runTrace = null;
    state.draft = null; state.draftSequence = 0; state.draftTokens = [];
    $('draft-text').replaceChildren(); renderDraftPanel();
    state.candidates = []; state.selected = -1; state.tab = 'best'; state.runElapsed = 0;
    $('output-empty').hidden = false; $('output-content').hidden = true;
    $('progress').hidden = true; $('run-error').hidden = true;
    $('source-preview').hidden = true;
    renderResults();
  }

  function setTarget(target) {
    if (!state.map) return;
    hideTooltip();
    abandonRun(); resetOutput();
    const t = state.transform;
    state.target = [Math.max(t.minX, Math.min(t.maxX, target[0])), Math.max(t.minY, Math.min(t.maxY, target[1]))];
    $('plot-instruction').hidden = true;
    $('target-coordinates').textContent = coordinateText(state.target);
    $('target-coordinates').parentElement.classList.add('has-target');
    renderMap(); renderNeighbors(); updateControls();
    announce(`Target selected at ${coordinateText(state.target)}.`);
  }

  function renderTarget() {
    const layer = $('target-layer'); layer.replaceChildren();
    if (!state.target || !state.transform) return;
    const [x, y] = state.transform.toScreen(...state.target);
    appendSVG(layer, 'circle', { cx: x, cy: y, r: 18, class: 'target-halo' });
    appendSVG(layer, 'circle', { cx: x, cy: y, r: 6, class: 'target-mark' });
    appendSVG(layer, 'path', { d: `M ${x - 12} ${y} H ${x - 3} M ${x + 3} ${y} H ${x + 12} M ${x} ${y - 12} V ${y - 3} M ${x} ${y + 3} V ${y + 12}`, class: 'target-mark' });
  }

  function bestIndex() {
    let result = -1;
    state.candidates.forEach((candidate, index) => { if (result < 0 || candidate.distance < state.candidates[result].distance) result = index; });
    return result;
  }

  function selectedIndex() { return state.selected >= 0 && state.selected < state.candidates.length ? state.selected : bestIndex(); }

  function renderResults() {
    const layer = $('result-layer'), trails = $('trail-layer');
    layer.replaceChildren(); trails.replaceChildren();
    if (!state.transform || !state.target || !state.candidates.length) return;
    const selected = selectedIndex();
    const [tx, ty] = state.transform.toScreen(...state.target);
    state.candidates.forEach((candidate, index) => {
      if (!finite(candidate.x) || !finite(candidate.y)) return;
      const [x, y] = state.transform.toScreen(candidate.x, candidate.y);
      if (index === selected) {
        appendSVG(trails, 'line', { x1: tx, y1: ty, x2: x, y2: y, class: 'result-line' });
        if (Math.hypot(x - tx, y - ty) > 55) {
          const lx = (x + tx) / 2, ly = (y + ty) / 2;
          appendSVG(trails, 'rect', { x: lx - 23, y: ly - 8, width: 46, height: 16, rx: 2, class: 'result-label-bg' });
          appendSVG(trails, 'text', { x: lx, y: ly + 3, 'text-anchor': 'middle', class: 'plot-distance' }, distanceText(candidate.distance));
        }
      }
      const g = appendSVG(layer, 'g', { tabindex: '0', role: 'button', 'aria-label': `Read candidate ${index + 1}, distance ${distanceText(candidate.distance)}` });
      appendSVG(g, 'circle', { cx: x, cy: y, r: index === selected ? 8 : 5.5, class: 'result-dot', fill: index === selected ? '#b55c3d' : '#d5a890' });
      appendSVG(g, 'text', { x: x + 10, y: y - 9, class: 'result-number' }, String(index + 1).padStart(2, '0'));
      const select = event => { event.stopPropagation(); state.selected = index; state.tab = 'best'; renderOutput(); renderResults(); };
      g.addEventListener('click', select);
      g.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(event); } });
    });
  }

  function updateSaveControl() {
    const candidate = state.candidates[selectedIndex()];
    const saved = candidate && state.saved.some(item => item.source.run_id === state.runId && item.source.candidate_index === candidate.sourceIndex);
    $('save-paragraph').disabled = !candidate || !state.runId || state.busy || state.saving || saved;
    $('save-paragraph').textContent = saved ? 'Saved to library' : state.saving ? 'Saving…' : 'Save paragraph';
    $('save-help').textContent = state.busy ? 'Finish or stop the search to save a measured paragraph.' : 'Keep this paragraph and its measured position on this host.';
  }

  function libraryError(message) {
    $('library-error').textContent = message || '';
    $('library-error').hidden = !message;
  }

  async function loadSaved() {
    try {
      const data = await api(`/api/saved?map_id=${encodeURIComponent(state.map.map_id)}`);
      if (data.map_id !== state.map.map_id) throw new Error('The saved library belongs to another map. Reload this page.');
      state.saved = data.paragraphs.filter(item => item.map_id === state.map.map_id && finite(item.x) && finite(item.y));
      libraryError(); renderSavedList(); renderSavedPoints(); updateSaveControl();
    } catch (error) { libraryError(`Could not load saved paragraphs. ${error.message}`); }
  }

  async function saveParagraph() {
    const candidate = state.candidates[selectedIndex()];
    if (!candidate || $('save-paragraph').disabled) return;
    const body = { map_id: state.map.map_id, run_id: state.runId, candidate_index: candidate.sourceIndex };
    state.saving = true; updateSaveControl(); libraryError();
    try {
      const saved = await api('/api/saved', { method: 'POST', body: JSON.stringify(body) });
      state.saved = [saved, ...state.saved.filter(item => item.id !== saved.id)];
      renderSavedList(); renderSavedPoints();
      announce('Paragraph saved on this host with its measured map position.');
    } catch (error) { libraryError(`Could not save this paragraph. ${error.message}`); }
    finally { state.saving = false; updateSaveControl(); }
  }

  function openSaved(item) {
    const details = $(`saved-${item.id}`);
    if (!details) return;
    details.open = true;
    details.querySelector('summary').focus({ preventScroll: true });
    details.scrollIntoView({ behavior: 'auto', block: 'nearest' });
    announce(`Saved paragraph: ${item.title}. Position ${coordinateText([item.x, item.y])}.`);
  }

  function renderSavedList() {
    const list = $('saved-list'); list.replaceChildren();
    $('saved-count').textContent = String(state.saved.length);
    if (!state.saved.length) list.append(element('p', { class: 'empty-context' }, 'Save a generated paragraph to keep its text and measured position here.'));
    state.saved.forEach(item => {
      const details = element('details', { class: 'saved-paragraph', id: `saved-${item.id}` });
      details.append(element('summary', {}, item.title));
      details.append(element('p', { class: 'saved-text' }, item.text));
      const when = new Date(item.created_at).toLocaleString();
      details.append(element('p', { class: 'saved-metadata' }, `Position ${coordinateText([item.x, item.y])} · target ${coordinateText(item.target)} · distance ${distanceText(item.distance)}\n${item.source.generator} · ${when}`));
      const actions = element('div', { class: 'saved-actions' });
      const locate = element('button', { type: 'button', class: 'text-button' }, 'Locate on map');
      locate.addEventListener('click', () => {
        $('show-saved').checked = true;
        setView(state.transform.scale, item.x, item.y); renderMap();
        $('map').focus({ preventScroll: true }); $('map').scrollIntoView({ block: 'nearest' });
        announce(`View centered on saved paragraph ${item.title}.`);
      });
      const remove = element('button', { type: 'button', class: 'text-button', 'aria-label': `Remove saved paragraph ${item.title}` }, 'Remove');
      remove.addEventListener('click', async () => {
        remove.disabled = true; libraryError();
        try {
          await api(`/api/saved/${encodeURIComponent(item.id)}?map_id=${encodeURIComponent(state.map.map_id)}`, { method: 'DELETE' });
          state.saved = state.saved.filter(record => record.id !== item.id);
          renderSavedList(); renderSavedPoints(); updateSaveControl();
          $('show-saved').focus({ preventScroll: true }); announce('Saved paragraph removed.');
        } catch (error) { libraryError(error.message); remove.disabled = false; }
      });
      actions.append(locate, remove); details.append(actions); list.append(details);
    });
  }

  function renderSavedPoints() {
    const layer = $('saved-layer'); layer.replaceChildren();
    if (!state.transform || !$('show-saved').checked) return;
    state.saved.forEach(item => {
      const [x, y] = state.transform.toScreen(item.x, item.y);
      if (x < plot.left || x > plot.left + plot.width || y < plot.top || y > plot.top + plot.height) return;
      const group = appendSVG(layer, 'g', { tabindex: '0', role: 'button', class: 'saved-point', 'aria-label': `Read saved paragraph: ${item.title}` });
      appendSVG(group, 'path', { d: `M ${x} ${y - 7} L ${x + 7} ${y} L ${x} ${y + 7} L ${x - 7} ${y} Z` });
      appendSVG(group, 'title', {}, `Saved: ${item.title}`);
      group.addEventListener('click', event => { event.stopPropagation(); openSaved(item); });
      group.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); event.stopPropagation(); openSaved(item); }
      });
    });
  }

  function renderDraftPanel() {
    const visible = state.status?.generator?.draft_streaming === true && (state.busy || state.draft !== null);
    $('draft-panel').hidden = !visible;
    $('draft-panel').classList.toggle('waiting', visible && state.busy);
    if (!visible) return;
    if (!state.candidates.length) $('output-empty').hidden = true;
    const draft = state.draft;
    if (!draft) {
      $('draft-step').textContent = '';
      $('draft-status').textContent = 'Waiting for the first native diffusion pass…';
      return;
    }
    const step = Number.isInteger(draft.step) ? draft.step + 1 : null;
    const total = Number.isInteger(draft.num_steps) && draft.num_steps > 0 ? draft.num_steps : null;
    const phase = (draft.phase || 'diffusion').replaceAll('_', ' ');
    const basin = Number.isInteger(draft.basin_id) ? ` · basin ${draft.basin_id + 1}` : '';
    $('draft-step').textContent = `${phase}${basin}${step !== null ? ` · step ${total ? Math.min(step, total) : step}${total ? ` / ${total}` : ''}` : ''}`;
    $('draft-status').textContent = state.busy ? draft.terminal ? 'Draft pass finished. Waiting for canonical measurement…' : 'Waiting for the next model revision…' : state.candidates.length ? 'Draft stream finished. Scored paragraphs are shown below.' : 'Draft stream finished. No canonically scored paragraph is available.';
  }

  function applyDraft(run) {
    if (!run.draft || !Number.isInteger(run.draft_sequence) || run.draft_sequence <= state.draftSequence) return;
    const tokens = String(run.draft.text || '').split(/(\s+)/);
    const previous = state.draftTokens;
    const content = document.createDocumentFragment();
    tokens.forEach((word, index) => {
      if (previous.length && word.trim() && word !== previous[index]) content.append(element('mark', { class: 'draft-change' }, word));
      else content.append(document.createTextNode(word));
    });
    $('draft-text').replaceChildren(content);
    state.draft = run.draft; state.draftSequence = run.draft_sequence; state.draftTokens = tokens;
  }

  function candidateHeading(candidate, index, isBest) {
    const heading = element('div', { class: 'result-heading' });
    heading.append(element('span', { class: 'result-label' }, `CANDIDATE ${String(index + 1).padStart(2, '0')}${isBest ? ' · BEST MATCH' : ''}`));
    const distance = element('span', { class: 'distance-chip' });
    distance.append(element('span', { 'aria-hidden': 'true' }, '↔'), element('strong', {}, distanceText(candidate.distance)), element('span', {}, 'from target'));
    heading.append(distance);
    return heading;
  }

  function renderOutput() {
    const count = state.candidates.length;
    $('output-empty').hidden = count > 0; $('output-content').hidden = count === 0;
    if (!count) { renderDraftPanel(); return; }
    const index = selectedIndex(), best = bestIndex(), candidate = state.candidates[index];
    updateSaveControl();
    $('candidate-total').textContent = String(count);
    $('result-label').textContent = `CANDIDATE ${String(index + 1).padStart(2, '0')}${index === best ? ' · BEST MATCH' : ''}`;
    $('result-distance').textContent = distanceText(candidate.distance);
    $('result-text').textContent = candidate.text;
    $('result-position').textContent = `ACTUAL POSITION  ${coordinateText([candidate.x, candidate.y])}`;
    $('result-time').textContent = `${format(candidate.elapsed_seconds, 1)}s`;
    const trajectories = state.runTrace?.trajectories;
    const trajectory = Array.isArray(trajectories) ? trajectories.find(item => item.seed === candidate.seed) : null;
    const guidedSteps = trajectory?.guidance?.steps;
    const passes = trajectory?.runtime?.passes;
    const executed = state.runTrace?.executed_work;
    const searchTotals = state.status?.generator?.refinement && executed && Number.isInteger(executed.native_passes);
    const hasCompute = searchTotals || Array.isArray(guidedSteps) && Number.isInteger(passes) && passes >= 0;
    $('result-compute').hidden = !hasCompute;
    if (searchTotals) {
      $('result-compute').textContent = `Search totals: ${executed.encoder_updates} encoder updates · ${executed.native_passes} diffusion passes${executed.trajectories_without_runtime_receipt ? ' · some failed work unmeasured' : ''}`;
    } else if (hasCompute) {
      const updates = guidedSteps.reduce((total, step) => total + (Array.isArray(step.updates) ? step.updates.length : 0), 0);
      $('result-compute').textContent = `${updates} encoder ${updates === 1 ? 'update' : 'updates'} · ${passes} diffusion ${passes === 1 ? 'pass' : 'passes'}`;
    }
    const all = $('all-view'); all.replaceChildren();
    state.candidates.forEach((item, itemIndex) => {
      const article = element('article', { class: 'all-candidate' });
      const button = element('button', { type: 'button', 'aria-label': `Select candidate ${itemIndex + 1}` });
      button.append(candidateHeading(item, itemIndex, itemIndex === best), element('p', {}, item.text));
      button.addEventListener('click', () => { state.selected = itemIndex; state.tab = 'best'; renderOutput(); renderResults(); });
      article.append(button); all.append(article);
    });
    setTab(state.tab);
  }

  function setTab(tab) {
    state.tab = tab;
    const best = tab === 'best';
    $('tab-best').classList.toggle('active', best); $('tab-best').setAttribute('aria-selected', String(best));
    $('tab-all').classList.toggle('active', !best); $('tab-all').setAttribute('aria-selected', String(!best));
    $('best-view').hidden = !best; $('all-view').hidden = best;
    $('copy').hidden = !best;
  }

  function updateElapsed() {
    const elapsed = state.busy ? Math.max(state.runElapsed, (performance.now() - state.started) / 1000) : state.runElapsed;
    $('elapsed').textContent = `${format(elapsed, 1)}s`;
  }

  function showRunError(message) { $('run-error').textContent = message; $('run-error').hidden = false; }

  function applyRun(run) {
    if (Number.isInteger(run.seed)) $('detail-seed').textContent = String(run.seed);
    applyDraft(run);
    state.runTrace = run.trace || null;
    state.candidates = Array.isArray(run.candidates) ? run.candidates.map((c, sourceIndex) => ({ ...c, sourceIndex })).filter(c => typeof c.text === 'string' && finite(c.x) && finite(c.y) && finite(c.distance)) : [];
    state.runElapsed = finite(run.elapsed_seconds) ? run.elapsed_seconds : state.runElapsed;
    const count = state.candidates.length;
    const running = run.status === 'running';
    $('progress').classList.toggle('running', running);
    $('progress').hidden = false;
    $('progress-fill').style.width = `${Math.min(100, count / state.requested * 100)}%`;
    $('progress-label').textContent = running ? count ? `${count} of ${state.requested} measured · finding the next paragraph` : 'Writing the first paragraph locally' : run.status === 'cancelled' ? `Search stopped · ${count} measured` : run.status === 'error' ? 'Search interrupted' : `${count} ${count === 1 ? 'paragraph' : 'paragraphs'} measured · closest selected`;
    if (state.status?.generator?.refinement) {
      const reasons = { epsilon_reached: 'Tolerance confirmed', budget_exhausted: 'Budget reached', restarts_exhausted: 'Restarts exhausted', stalled: 'Local search stalled', cancelled: 'Search stopped', no_valid_text: 'No valid paragraph' };
      const reason = run.trace?.stop_reason;
      const restarting = running && run.draft?.phase?.includes('coarse') && run.draft?.basin_id > 0;
      const restarts = run.trace?.restarts;
      const restartCount = Array.isArray(restarts) ? restarts.length : 0;
      $('progress-label').textContent = running ? restarting ? `${count} verified improvements · trying a fresh generation` : count ? `${count} verified improvements · refining actual text` : 'Generating and checking clean drafts' : `${reasons[reason] || 'Search finished'} · ${count} verified improvements${restartCount ? ` · ${restartCount} ${restartCount === 1 ? 'restart' : 'restarts'}` : ''}`;
    }
    if (run.error) showRunError(run.error);
    renderOutput();
    renderDraftPanel();
    renderResults();
    updateElapsed();
    return running;
  }

  function finishRun() {
    state.busy = false;
    window.clearInterval(state.tickTimer);
    $('progress').classList.remove('running');
    updateElapsed(); updateControls();
  }

  async function pollRun(id, epoch) {
    if (epoch !== state.epoch || state.runId !== id) return;
    try {
      const run = await api(`/api/runs/${encodeURIComponent(id)}`);
      if (epoch !== state.epoch || state.runId !== id) return;
      if (applyRun(run)) {
        state.runTimer = window.setTimeout(() => pollRun(id, epoch), 700);
      } else {
        finishRun();
        announce(run.status === 'complete' ? `Search complete. ${state.candidates.length} ${state.candidates.length === 1 ? 'paragraph' : 'paragraphs'} measured. Best distance ${format(state.candidates[bestIndex()]?.distance)}.` : run.status === 'cancelled' ? 'Search stopped.' : 'The local generation encountered an error.');
      }
    } catch (error) {
      if (epoch !== state.epoch) return;
      showRunError(`${error.message} Try generating again.`);
      void quietlyCancel(id);
      finishRun();
    }
  }

  async function generate() {
    if ($('generate').disabled) return;
    abandonRun(); resetOutput();
    const epoch = state.epoch;
    state.busy = true; state.started = performance.now(); state.requested = Number($('candidate-count').value);
    $('progress').hidden = false; $('progress').classList.add('running');
    $('progress-label').textContent = 'Starting local generation'; $('progress-fill').style.width = '3%';
    state.tickTimer = window.setInterval(updateElapsed, 100);
    updateControls(); updateElapsed();
    announce(`Generating ${state.requested} candidate ${state.requested === 1 ? 'paragraph' : 'paragraphs'} locally.`);
    try {
      const body = { map_id: state.map.map_id, target: [...state.target], candidates: state.requested, seed: freshSeed() };
      $('detail-seed').textContent = String(body.seed);
      if (state.status?.generator?.guided === true) body.guidance_updates = Number($('guidance-updates').value);
      if (state.status?.generator?.refinement === true) {
        body.refinement_epsilon = Number($('refinement-epsilon').value);
        body.refinement_seconds = Number($('refinement-seconds').value);
        body.refinement_restarts = Number($('refinement-restarts').value);
        body.refinement_restart_candidates = Number($('refinement-restart-candidates').value);
        if ($('refinement-unlimited').checked) {
          delete body.refinement_seconds;
          body.refinement_unlimited = true;
        }
      }
      const run = await api('/api/runs', { method: 'POST', body: JSON.stringify(body) });
      if (!run.id) throw new Error('The local worker did not return a run identifier.');
      if (epoch !== state.epoch) { void quietlyCancel(run.id); return; }
      state.runId = run.id;
      void pollRun(run.id, epoch);
    } catch (error) {
      if (epoch !== state.epoch) return;
      showRunError(error.message);
      $('progress-label').textContent = 'The search could not start';
      finishRun();
    }
  }

  async function cancel() {
    const id = state.runId, epoch = ++state.epoch;
    $('cancel').disabled = true;
    $('progress-label').textContent = 'Stopping the local search…';
    if (!id) {
      abandonRun();
      $('progress-label').textContent = 'Search stopped';
      finishRun(); $('cancel').disabled = false;
      announce('Search stopped.');
      return;
    }
    window.clearTimeout(state.runTimer);
    try {
      const run = await api(`/api/runs/${encodeURIComponent(id)}`, { method: 'DELETE' });
      if (epoch !== state.epoch) return;
      if (applyRun(run)) state.runTimer = window.setTimeout(() => pollRun(id, epoch), 700);
      else { finishRun(); announce('Search stopped. Completed candidates are kept.'); }
    } catch (error) {
      if (epoch === state.epoch) { showRunError(error.message); state.runTimer = window.setTimeout(() => pollRun(id, epoch), 700); }
    } finally { $('cancel').disabled = false; }
  }

  function eventPoint(event) {
    const point = $('map').createSVGPoint(); point.x = event.clientX; point.y = event.clientY;
    const matrix = $('map').getScreenCTM();
    if (!matrix) return null;
    const local = point.matrixTransform(matrix.inverse());
    return [local.x, local.y];
  }

  function insidePlot(point) {
    return point && point[0] >= plot.left && point[0] <= plot.left + plot.width && point[1] >= plot.top && point[1] <= plot.top + plot.height;
  }

  function beginGesture() {
    const points = [...state.pointers.values()];
    if (!points.length) { state.gesture = null; return; }
    const center = points.length > 1 ? [(points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2] : points[0];
    state.gesture = { center, scale: state.transform.scale, cx: state.transform.cx, cy: state.transform.cy, anchor: state.transform.toWorld(...center), distance: points.length > 1 ? Math.hypot(points[0][0] - points[1][0], points[0][1] - points[1][1]) : 0, moved: state.suppressClick };
  }

  $('map').addEventListener('pointerdown', event => {
    const point = eventPoint(event);
    if (!state.map || !insidePlot(point) || event.button !== 0) return;
    if (!state.pointers.size) state.suppressClick = false;
    state.pointers.set(event.pointerId, point); beginGesture();
    if (state.pointers.size > 1 || state.panMode) state.suppressClick = true;
  });
  $('map').addEventListener('pointermove', event => {
    if (!state.pointers.has(event.pointerId) || !state.gesture) return;
    const point = eventPoint(event); if (!point) return;
    state.pointers.set(event.pointerId, point);
    const points = [...state.pointers.values()], g = state.gesture;
    const center = points.length > 1 ? [(points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2] : point;
    if (!g.moved && points.length < 2 && Math.hypot(center[0] - g.center[0], center[1] - g.center[1]) < 5) return;
    g.moved = true; state.suppressClick = true;
    $('map').setPointerCapture(event.pointerId);
    $('map').classList.add('is-panning'); hideTooltip();
    if (points.length > 1 && g.distance > 0) {
      const distance = Math.hypot(points[0][0] - points[1][0], points[0][1] - points[1][1]);
      const scale = Math.max(state.baseScale / 8, Math.min(state.baseScale * 100, g.scale * distance / g.distance));
      const { sx, sy } = state.transform;
      setView(scale, g.anchor[0] - (center[0] - sx) / scale, g.anchor[1] + (center[1] - sy) / scale);
    } else setView(g.scale, g.cx - (center[0] - g.center[0]) / g.scale, g.cy + (center[1] - g.center[1]) / g.scale);
    renderMap();
  });
  function endPointer(event) {
    if (!state.pointers.has(event.pointerId)) return;
    state.pointers.delete(event.pointerId);
    if ($('map').hasPointerCapture(event.pointerId)) $('map').releasePointerCapture(event.pointerId);
    $('map').classList.remove('is-panning'); beginGesture();
  }
  $('map').addEventListener('pointerup', endPointer);
  $('map').addEventListener('pointercancel', endPointer);
  $('map').addEventListener('pointerleave', event => { if (!$('map').hasPointerCapture(event.pointerId)) endPointer(event); });
  $('map').addEventListener('click', event => {
    if (state.suppressClick || state.panMode) { event.preventDefault(); event.stopImmediatePropagation(); state.suppressClick = false; }
  }, true);
  $('map').addEventListener('wheel', event => {
    const point = eventPoint(event);
    if (!state.map || !insidePlot(point)) return;
    event.preventDefault();
    const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 490 : 1);
    zoomAt(Math.exp(-Math.max(-200, Math.min(200, delta)) * (event.ctrlKey ? .008 : .002)), point);
  }, { passive: false });

  $('map').addEventListener('click', event => {
    if (!state.map) return;
    const svgPoint = $('map').createSVGPoint(); svgPoint.x = event.clientX; svgPoint.y = event.clientY;
    const screenMatrix = $('map').getScreenCTM(); if (!screenMatrix) return;
    const point = svgPoint.matrixTransform(screenMatrix.inverse());
    if (point.x < plot.left || point.x > plot.left + plot.width || point.y < plot.top || point.y > plot.top + plot.height) return;
    hideTooltip(); setTarget(state.transform.toWorld(point.x, point.y));
  });

  $('map').addEventListener('keydown', event => {
    if (event.target !== $('map') || !state.map) return;
    const delta = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[event.key];
    if (['+', '=', '-', '0'].includes(event.key)) {
      event.preventDefault();
      if (event.key === '0') frameMap(); else zoomAt(event.key === '-' ? 1 / 1.4 : 1.4);
      return;
    }
    if (!delta) return;
    event.preventDefault();
    const t = state.transform, target = state.target || [t.cx, t.cy];
    if (event.altKey) {
      setView(t.scale, t.cx + delta[0] * 45 / t.scale, t.cy + delta[1] * 45 / t.scale); renderMap();
      return;
    }
    const step = (event.shiftKey ? 12 : 3) / t.scale;
    setTarget([target[0] + delta[0] * step, target[1] + delta[1] * step]);
  });

  $('clear-target').addEventListener('click', () => {
    abandonRun(); state.target = null; resetOutput(); renderMap(); renderNeighbors();
    $('target-coordinates').textContent = 'Choose a point on the map';
    $('target-coordinates').parentElement.classList.remove('has-target');
    $('plot-instruction').hidden = !state.map;
    updateControls(); announce('Target cleared.');
  });
  $('generate').addEventListener('click', generate);
  $('zoom-in').addEventListener('click', () => zoomAt(1.4));
  $('zoom-out').addEventListener('click', () => zoomAt(1 / 1.4));
  $('fit-map').addEventListener('click', frameMap);
  $('pan-mode').addEventListener('click', () => {
    state.panMode = !state.panMode;
    $('pan-mode').setAttribute('aria-pressed', String(state.panMode));
    $('map').classList.toggle('pan-mode', state.panMode);
    announce(state.panMode ? 'Pan mode on. Drag the map. Turn pan mode off to select a target.' : 'Target mode on. Click to select a target, or drag to pan.');
  });
  $('save-paragraph').addEventListener('click', saveParagraph);
  $('refinement-unlimited').addEventListener('change', updateControls);
  $('refinement-restarts').addEventListener('change', updateControls);
  $('show-saved').addEventListener('change', renderSavedPoints);
  $('cancel').addEventListener('click', cancel);
  $('close-source').addEventListener('click', () => { $('source-preview').hidden = true; });
  $('tab-best').addEventListener('click', () => { state.selected = -1; renderOutput(); setTab('best'); renderResults(); });
  $('tab-all').addEventListener('click', () => setTab('all'));
  for (const id of ['tab-best', 'tab-all']) {
    $(id).addEventListener('keydown', event => {
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
      event.preventDefault();
      const next = id === 'tab-best' ? $('tab-all') : $('tab-best');
      next.focus(); next.click();
    });
  }
  $('copy').addEventListener('click', async () => {
    const candidate = state.candidates[selectedIndex()]; if (!candidate) return;
    try {
      await navigator.clipboard.writeText(candidate.text);
      $('copy').textContent = 'Copied'; announce('Paragraph copied.');
      window.setTimeout(() => { $('copy').textContent = 'Copy text'; }, 1800);
    } catch { announce('Copy is unavailable in this browser. Select the paragraph text to copy it.'); }
  });

  window.addEventListener('pagehide', () => {
    if (state.runId && state.busy) fetch(`/api/runs/${encodeURIComponent(state.runId)}`, { method: 'DELETE', keepalive: true }).catch(() => {});
  });
  void checkStatus();
})();
