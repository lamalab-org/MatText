/* ==========================================================================
   MatText — interactive figures
   Every number below comes straight from the manuscript's source data
   (artifact/Source_Data/*.zip) — nothing here is invented for effect.
   ========================================================================== */

(() => {
  'use strict';

  const svgNS = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs = {}, ns = svgNS) => {
    const node = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    return node;
  };

  /* ------------------------------------------------------------------ *
   * Figure 1 — coordinate / category / both toggle
   * ------------------------------------------------------------------ */
  (function fig1() {
    const svg = document.getElementById('fig1-svg');
    if (!svg) return;

    const POINTS = [
      { x: 0.8, y: 0.6, t: 'a' }, { x: 1.2, y: 1.4, t: 'b' }, { x: 1.8, y: 0.9, t: 'c' },
      { x: 2.3, y: 2.1, t: 'a' }, { x: 2.8, y: 1.6, t: 'c' }, { x: 3.2, y: 0.8, t: 'b' },
      { x: 1.5, y: 2.3, t: 'c' }, { x: 0.9, y: 1.9, t: 'a' }, { x: 2.6, y: 0.5, t: 'b' },
      { x: 3.1, y: 2.2, t: 'b' }, { x: 1.1, y: 1.1, t: 'a' }, { x: 2.0, y: 1.8, t: 'c' },
      { x: 2.9, y: 1.3, t: 'a' }, { x: 0.7, y: 2.5, t: 'b' }, { x: 1.6, y: 0.7, t: 'a' },
      { x: 2.4, y: 1.2, t: 'c' }, { x: 3.3, y: 1.9, t: 'b' }, { x: 1.3, y: 2.6, t: 'c' },
    ];

    const COL_X = { a: 130, b: 245, c: 360 };
    const READOUT = {
      coordinate:
        'Only <em>where</em>. Continuous positions, no labels attached. A geometry-aware model can use this directly — a language model has nothing to name.',
      category:
        'Only <em>what</em>. Discrete types, spatial layout thrown away. This is exactly what a language model reads natively.',
      both: 'Real data has both. The open question is whether a language model uses the coordinate half, or quietly discards it.',
    };

    const counts = { a: 0, b: 0, c: 0 };
    POINTS.forEach((p) => {
      p.coordX = 55 + p.x * 95;
      p.coordY = 262 - p.y * 80;
      p.catX = COL_X[p.t];
      p.catY = 56 + counts[p.t] * 38;
      counts[p.t] += 1;
    });

    const axes = el('g', { class: 'f1-aux', id: 'f1-axes' });
    axes.appendChild(el('line', { x1: 40, y1: 270, x2: 420, y2: 270, class: 'f1-axis' }));
    axes.appendChild(el('line', { x1: 40, y1: 270, x2: 40, y2: 30, class: 'f1-axis' }));
    const xLabel = el('text', { x: 424, y: 274, class: 'f1-label', fill: 'var(--ink-faint)' });
    xLabel.textContent = 'x';
    const yLabel = el('text', { x: 34, y: 24, class: 'f1-label', fill: 'var(--ink-faint)' });
    yLabel.textContent = 'y';
    axes.appendChild(xLabel);
    axes.appendChild(yLabel);
    svg.appendChild(axes);

    const colLabels = el('g', { class: 'f1-aux', id: 'f1-col-labels', style: 'opacity:0' });
    ['a', 'b', 'c'].forEach((t) => {
      const label = el('text', { x: COL_X[t], y: 286, class: 'f1-label', fill: 'var(--ink-faint)', 'text-anchor': 'middle' });
      label.textContent = 'type ' + t.toUpperCase();
      colLabels.appendChild(label);
    });
    svg.appendChild(colLabels);

    const dotsGroup = el('g');
    POINTS.forEach((p) => {
      const dot = el('circle', { class: 'f1-dot type-' + p.t, cx: p.coordX, cy: p.coordY, r: 6 });
      p.node = dot;
      dotsGroup.appendChild(dot);
    });
    svg.appendChild(dotsGroup);

    function setMode(mode) {
      POINTS.forEach((p) => {
        const useCat = mode === 'category';
        p.node.setAttribute('cx', useCat ? p.catX : p.coordX);
        p.node.setAttribute('cy', useCat ? p.catY : p.coordY);
        p.node.classList.toggle('gray', mode === 'coordinate');
      });
      axes.style.opacity = mode === 'category' ? '0' : '1';
      colLabels.style.opacity = mode === 'category' ? '1' : '0';

      document.querySelectorAll('.mode-btn').forEach((btn) => {
        btn.setAttribute('aria-pressed', String(btn.dataset.mode === mode));
      });
      const readout = document.getElementById('fig1-readout');
      if (readout) readout.innerHTML = READOUT[mode];
    }

    document.querySelectorAll('.mode-btn').forEach((btn) => {
      btn.addEventListener('click', () => setMode(btn.dataset.mode));
    });

    setMode('both');
  })();

  /* ------------------------------------------------------------------ *
   * Figure 2 — CC-Cliff, reproducing the paper's own CoC/CaC formula
   * (Methods, "Coordinate-category cliff"):
   *   CoC = Σ loss(α) for α∈{0,0.2,0.4}  − 3·loss(0.5)
   *   CaC = Σ loss(α) for α∈{0.6,0.8,1}  − 3·loss(0.5)
   * Each line below plots loss(α) − loss(0.5) — i.e. error in excess of
   * the balanced baseline — per representation, meaned across
   * representations then across datasets (matching plots/figure_2.py
   * exactly; linear, so summing/averaging commute). The left three
   * points sum to CoC, the right three to CaC, by construction; the
   * curve crosses zero at α = 0.5 for both models.
   * Source: Figure 2 data (artifact/Source_Data/MatText_Fig2_SourceData).
   * ------------------------------------------------------------------ */
  (function cliff() {
    const svg = document.getElementById('cliff-svg');
    const range = document.getElementById('alpha-range');
    if (!svg || !range) return;

    const ALPHAS = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0];
    // loss(α) − loss(0.5), averaged across representations then across
    // datasets (MatText: 6 datasets, matching the paper's Fig. 2 legend;
    // CoGN: 7 — CoGN has no separate per-representation axis to average).
    const LLM = [0.2984, 0.1563, 0.0451, 0.0, -0.0344, -0.0763, -0.0871];
    const GNN = [-0.0194, -0.0371, -0.0211, 0.0, 0.014, 0.0589, 0.1426];

    const STATS = {
      llm: { coc: 0.4998, cac: -0.1978, cliff: 0.6976 },
      gnn: { coc: -0.0777, cac: 0.2154, cliff: -0.2931 },
    };
    const statsEl = document.getElementById('cliff-stats');
    if (statsEl) {
      statsEl.innerHTML = `
        <span><b>MatText (LLM)</b> CoC ${STATS.llm.coc.toFixed(2)} · CaC ${STATS.llm.cac.toFixed(2)} · CC-Cliff ${STATS.llm.cliff >= 0 ? '+' : ''}${STATS.llm.cliff.toFixed(2)}</span>
        <span><b>CoGN (GNN)</b> CoC ${STATS.gnn.coc.toFixed(2)} · CaC ${STATS.gnn.cac.toFixed(2)} · CC-Cliff ${STATS.gnn.cliff >= 0 ? '+' : ''}${STATS.gnn.cliff.toFixed(2)}</span>
      `;
    }

    const M = { left: 50, right: 16, top: 18, bottom: 40 };
    const W = 640, H = 300;
    const plotW = W - M.left - M.right;
    const plotH = H - M.top - M.bottom;

    const allV = LLM.concat(GNN);
    const rawMin = Math.min(...allV);
    const rawMax = Math.max(...allV);
    const pad = (rawMax - rawMin) * 0.15;
    const yMin = rawMin - pad;
    const yMax = rawMax + pad;

    const xAt = (a) => M.left + a * plotW;
    const yAt = (v) => M.top + ((yMax - v) / (yMax - yMin)) * plotH;

    function toPoints(series) {
      return ALPHAS.map((a, i) => ({ x: xAt(a), y: yAt(series[i]) }));
    }

    function smoothPath(pts) {
      let d = `M${pts[0].x},${pts[0].y}`;
      for (let i = 0; i < pts.length - 1; i++) {
        const p0 = pts[i === 0 ? i : i - 1];
        const p1 = pts[i];
        const p2 = pts[i + 1];
        const p3 = pts[i + 2 < pts.length ? i + 2 : i + 1];
        const c1x = p1.x + (p2.x - p0.x) / 6;
        const c1y = p1.y + (p2.y - p0.y) / 6;
        const c2x = p2.x - (p3.x - p1.x) / 6;
        const c2y = p2.y - (p3.y - p1.y) / 6;
        d += ` C${c1x},${c1y} ${c2x},${c2y} ${p2.x},${p2.y}`;
      }
      return d;
    }

    function interp(series, alpha) {
      for (let i = 0; i < ALPHAS.length - 1; i++) {
        if (alpha >= ALPHAS[i] && alpha <= ALPHAS[i + 1]) {
          const t = (alpha - ALPHAS[i]) / (ALPHAS[i + 1] - ALPHAS[i]);
          return series[i] + t * (series[i + 1] - series[i]);
        }
      }
      return series[series.length - 1];
    }

    // zero line = the α=0.5 baseline itself (both curves cross it by construction)
    svg.appendChild(el('line', { x1: M.left, y1: yAt(0), x2: W - M.right, y2: yAt(0), class: 'cliff-grid' }));
    svg.appendChild(el('line', { x1: M.left, y1: H - M.bottom, x2: W - M.right, y2: H - M.bottom, class: 'cliff-axis' }));
    svg.appendChild(el('line', { x1: M.left, y1: M.top, x2: M.left, y2: H - M.bottom, class: 'cliff-axis' }));

    const yLbl = el('text', { x: 8, y: 14, class: 'cliff-tick' });
    yLbl.textContent = 'error vs. α=0.5 baseline';
    svg.appendChild(yLbl);

    [yMax, 0, yMin].forEach((v) => {
      const t = el('text', { x: M.left - 8, y: yAt(v) + 4, class: 'cliff-tick', 'text-anchor': 'end' });
      t.textContent = (v > 0 ? '+' : '') + v.toFixed(2);
      svg.appendChild(t);
    });

    [0, 0.5, 1].forEach((a) => {
      const t = el('text', { x: xAt(a), y: H - M.bottom + 22, class: 'cliff-tick', 'text-anchor': a === 0 ? 'start' : a === 1 ? 'end' : 'middle' });
      t.textContent = 'α = ' + a.toFixed(1);
      svg.appendChild(t);
    });

    svg.appendChild(el('path', { d: smoothPath(toPoints(LLM)), class: 'cliff-line-llm' }));
    svg.appendChild(el('path', { d: smoothPath(toPoints(GNN)), class: 'cliff-line-gnn' }));

    const markerLine = el('line', { x1: xAt(0.5), y1: M.top, x2: xAt(0.5), y2: H - M.bottom, class: 'cliff-marker-line' });
    const markerDotLLM = el('circle', { cx: xAt(0.5), cy: yAt(interp(LLM, 0.5)), r: 6, class: 'cliff-dot' });
    const markerDotGNN = el('circle', { cx: xAt(0.5), cy: yAt(interp(GNN, 0.5)), r: 5, fill: 'var(--dark-coord)', stroke: 'var(--dark-bg)', 'stroke-width': 1.5 });
    svg.appendChild(markerLine);
    svg.appendChild(markerDotGNN);
    svg.appendChild(markerDotLLM);

    const valueOut = document.getElementById('alpha-value');
    const message = document.getElementById('cliff-message');
    const verdict = document.getElementById('cliff-verdict');

    function update() {
      const a = Number(range.value);
      const x = xAt(a);
      const llmV = interp(LLM, a);
      const gnnV = interp(GNN, a);

      markerLine.setAttribute('x1', x);
      markerLine.setAttribute('x2', x);
      markerDotLLM.setAttribute('cx', x);
      markerDotLLM.setAttribute('cy', yAt(llmV));
      markerDotGNN.setAttribute('cx', x);
      markerDotGNN.setAttribute('cy', yAt(gnnV));

      valueOut.textContent = `α = ${a.toFixed(2)}`;

      if (a < 0.25) {
        message.textContent = 'Near-pure geometry. The language model sits well above its own balanced-case error; the GNN sits at or below it.';
      } else if (a > 0.75) {
        message.textContent = 'Near-pure composition. This is exactly what language models are built to read — the GNN is the one paying the cost here.';
      } else {
        message.textContent = 'Near the balanced midpoint (α = 0.5) — both curves are defined to cross zero here.';
      }
      const fmt = (v) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}`;
      verdict.textContent = `at α = ${a.toFixed(2)}: LLM ${fmt(llmV)} · GNN ${fmt(gnnV)} (error vs. their own α=0.5 baseline)`;
    }

    range.addEventListener('input', update);
    update();
  })();

  /* ------------------------------------------------------------------ *
   * Figure 5 — representation comparison (30K BERT results, 5-fold RMSE)
   * ------------------------------------------------------------------ */
  (function representations() {
    const svg = document.getElementById('rep-svg');
    const preview = document.getElementById('rep-preview');
    if (!svg || !preview) return;

    const GROUP_COLOR = { compositional: '#e76f51', local: '#c1121f', geometric: '#79155b' };
    const REP_OVERRIDE = { robocrys: '#6c757d' };

    const REP_ORDER = [
      { rep: 'composition', label: 'Composition', group: 'compositional' },
      { rep: 'atom_sequences', label: 'Atom Seq.', group: 'compositional' },
      { rep: 'atom_sequences_plusplus', label: 'Atom Seq.++', group: 'compositional' },
      { rep: 'robocrys', label: 'Robocrys', group: 'local' },
      { rep: 'local_env', label: 'Local-Env', group: 'local' },
      { rep: 'slices', label: 'SLICES', group: 'local' },
      { rep: 'zmatrix', label: 'Z-Matrix', group: 'geometric' },
      { rep: 'cif_symmetrized', label: 'CIF Sym.', group: 'geometric' },
      { rep: 'cif_p1', label: 'CIF P₁', group: 'geometric' },
      { rep: 'crystal_text_llm', label: 'Crys.-Text-LLM', group: 'geometric' },
    ];

    const VALUES = {
      gvrh: { composition: 0.1918, atom_sequences: 0.1765, atom_sequences_plusplus: 0.1718, robocrys: 0.1632, local_env: 0.1623, slices: 0.1516, zmatrix: 0.165, cif_symmetrized: 0.1705, cif_p1: 0.1626, crystal_text_llm: 0.1546 },
      kvrh: { composition: 0.1886, atom_sequences: 0.17, atom_sequences_plusplus: 0.1738, robocrys: 0.1549, local_env: 0.1578, slices: 0.1465, zmatrix: 0.1613, cif_symmetrized: 0.1706, cif_p1: 0.1643, crystal_text_llm: 0.1507 },
      perovskites: { composition: 0.5707, atom_sequences: 0.5595, atom_sequences_plusplus: 0.3281, robocrys: 0.1571, local_env: 0.104, slices: 0.0999, zmatrix: 0.1068, cif_symmetrized: 0.114, cif_p1: 0.1078, crystal_text_llm: 0.0966 },
    };

    const PREVIEWS = {
      composition: { name: 'Composition', desc: 'Just the chemical formula. No structure at all.', snippet: 'Sr1 Ti1 O3' },
      atom_sequences: { name: 'Atom Sequence', desc: 'One element symbol per atom, in cell order — still no coordinates.', snippet: 'Sr Ti O O O' },
      atom_sequences_plusplus: { name: 'Atom Sequence ++', desc: 'Adds space-group and lattice-system hints, but no explicit coordinates.', snippet: 'Sr Ti O O O\ncubic · Pm-3m' },
      robocrys: { name: 'Robocrystallographer', desc: 'A natural-language paragraph describing local bonding — generated by Robocrystallographer, not a raw file format.', snippet: 'SrTiO3 is Perovskite structured.\nTi is bonded to six equivalent O\natoms to form corner-sharing\nTiO6 octahedra.' },
      local_env: { name: 'Local-Env', desc: "Each atom's coordination environment: neighbor count and geometry type — not raw positions.", snippet: 'Sr: 12-coordinate, cuboctahedral\nTi: 6-coordinate, octahedral' },
      slices: { name: 'SLICES', desc: 'A bond graph — which atoms connect to which. No coordinates anywhere in the string.', snippet: 'Sr Ti O O O\n0 1 o o o\n0 2 + o o o …' },
      zmatrix: { name: 'Z-Matrix', desc: 'Each atom placed relative to earlier atoms by distance, angle, dihedral.', snippet: 'Ti  0.500 0.500 0.500\nO   1.953  90.0   0.0\nO   1.953  90.0  90.0' },
      cif_symmetrized: { name: 'CIF, symmetrized', desc: 'The crystallographic file format: unit cell + space group, one atom per symmetry site.', snippet: "_symmetry_space_group_name_H-M 'Pm-3m'\nSr1 Sr 1a 0.0 0.0 0.0\nTi1 Ti 1b 0.5 0.5 0.5\nO1  O  3d 0.5 0.5 0.0" },
      cif_p1: { name: 'CIF, P₁', desc: 'Symmetry expanded — every atom listed explicitly with full 3D fractional coordinates.', snippet: 'Sr1 Sr 0.000 0.000 0.000\nTi1 Ti 0.500 0.500 0.500\nO1  O  0.500 0.500 0.000\nO2  O  0.500 0.000 0.500\nO3  O  0.000 0.500 0.500' },
      crystal_text_llm: { name: 'Crystal-Text-LLM', desc: 'A compact generation-oriented layout: lattice lengths, angles, then fractional coordinates.', snippet: '3.9 3.9 3.9\n90 90 90\nSr 0.0 0.0 0.0\nTi 0.5 0.5 0.5\nO 0.5 0.5 0.0' },
    };

    const PROP_LABEL = { gvrh: 'shear modulus (μ)', kvrh: 'bulk modulus (K)', perovskites: 'perovskite Eꜰ' };

    let currentProp = 'gvrh';
    let currentRep = 'crystal_text_llm';

    function renderPreview(rep, prop) {
      const meta = PREVIEWS[rep];
      const value = VALUES[prop][rep];
      preview.innerHTML = `
        <p class="rep-preview-label">what the model reads</p>
        <p class="rep-preview-name">${meta.name}</p>
        <p class="rep-preview-desc">${meta.desc}</p>
        <pre>${meta.snippet}</pre>
        <p class="rep-preview-foot">RMSE, ${PROP_LABEL[prop]}: <b style="color:var(--ink)">${value.toFixed(3)}</b> · illustrative preview, SrTiO₃</p>
      `;
    }

    function renderChart(prop) {
      svg.innerHTML = '';
      const M = { left: 42, right: 10, top: 16, bottom: 96 };
      const W = 560, H = 340;
      const plotW = W - M.left - M.right;
      const plotH = H - M.top - M.bottom;
      const slot = plotW / REP_ORDER.length;
      const barW = slot * 0.58;

      const vals = REP_ORDER.map((r) => VALUES[prop][r.rep]);
      const maxV = Math.max(...vals) * 1.12;
      const y0 = M.top + plotH;
      const yAt = (v) => y0 - (v / maxV) * plotH;

      svg.appendChild(el('line', { x1: M.left, y1: y0, x2: W - M.right, y2: y0, class: 'rep-axis' }));

      REP_ORDER.forEach((r, i) => {
        const v = VALUES[prop][r.rep];
        const cx = M.left + slot * i + slot / 2;
        const barH = y0 - yAt(v);
        const color = REP_OVERRIDE[r.rep] || GROUP_COLOR[r.group];

        const group = el('g', { class: 'rep-bar', tabindex: '0', role: 'button', 'aria-label': `${r.label}, RMSE ${v.toFixed(3)}` });
        const rect = el('rect', {
          x: cx - barW / 2, y: yAt(v), width: barW, height: barH,
          fill: color, opacity: r.rep === currentRep ? 1 : 0.68,
          stroke: r.rep === currentRep ? 'var(--ink)' : 'none', 'stroke-width': 1.5,
        });
        group.appendChild(rect);

        const val = el('text', { x: cx, y: yAt(v) - 7, class: 'rep-bar-value', 'text-anchor': 'middle' });
        val.textContent = v.toFixed(3);
        group.appendChild(val);

        const label = el('text', { x: cx, y: y0 + 14, class: 'rep-bar-label', 'text-anchor': 'end', transform: `rotate(-42 ${cx} ${y0 + 14})` });
        label.textContent = r.label;
        group.appendChild(label);

        const select = () => {
          currentRep = r.rep;
          renderChart(prop);
          renderPreview(r.rep, prop);
        };
        group.addEventListener('click', select);
        group.addEventListener('keypress', (e) => { if (e.key === 'Enter') select(); });

        svg.appendChild(group);
      });
    }

    document.querySelectorAll('.tab-btn[data-prop]').forEach((tab) => {
      tab.addEventListener('click', () => {
        currentProp = tab.dataset.prop;
        document.querySelectorAll('.tab-btn[data-prop]').forEach((t) => t.setAttribute('aria-pressed', String(t === tab)));
        renderChart(currentProp);
        renderPreview(currentRep, currentProp);
      });
    });

    renderChart(currentProp);
    renderPreview(currentRep, currentProp);
  })();

  /* ------------------------------------------------------------------ *
   * Figure 6 — scaling (mean % change in RMSE across representations)
   * ------------------------------------------------------------------ */
  (function scaling() {
    const grid = document.getElementById('scale-grid');
    if (!grid) return;

    const DATA = {
      dataset: {
        xs: ['30K', '100K', '300K', '2M'],
        series: {
          'Bulk modulus (K)': [0, -1.22, -4.79, -7.1],
          'Shear modulus (μ)': [0, -1.58, -3.63, -8.18],
          'Perovskite Eꜰ': [0, -2.65, -4.16, -3.42],
        },
      },
      model: {
        xs: ['7B', '13B', '70B'],
        series: {
          'Bulk modulus (K)': [0, -3.52, -2.81],
          'Shear modulus (μ)': [0, -1.63, -2.49],
          'Perovskite Eꜰ': [0, -8.25, -15.71],
        },
      },
    };

    function render(mode) {
      const { xs, series } = DATA[mode];
      grid.innerHTML = '';

      Object.entries(series).forEach(([label, values]) => {
        const cell = document.createElement('div');
        cell.className = 'scale-cell';

        const W = 200, H = 120;
        const M = { left: 8, right: 8, top: 10, bottom: 22 };
        const plotW = W - M.left - M.right;
        const plotH = H - M.top - M.bottom;
        const allVals = values.concat(0);
        const lo = Math.min(...allVals) * 1.2;
        const hi = Math.max(...allVals, 2);
        const xAt = (i) => M.left + (i / (xs.length - 1)) * plotW;
        const yAt = (v) => M.top + (1 - (v - lo) / (hi - lo)) * plotH;

        const svg = el('svg', { viewBox: `0 0 ${W} ${H}` });
        svg.appendChild(el('line', { x1: M.left, x2: W - M.right, y1: yAt(0), y2: yAt(0), class: 'scale-zero' }));

        const pts = values.map((v, i) => `${xAt(i)},${yAt(v)}`).join(' ');
        svg.appendChild(el('polyline', { points: pts, class: 'scale-line' }));
        values.forEach((v, i) => svg.appendChild(el('circle', { cx: xAt(i), cy: yAt(v), r: 3, class: 'scale-dot' })));

        xs.forEach((x, i) => {
          const t = el('text', { x: xAt(i), y: H - 6, class: 'scale-tick', 'text-anchor': i === 0 ? 'start' : i === xs.length - 1 ? 'end' : 'middle' });
          t.textContent = x;
          svg.appendChild(t);
        });

        const last = values[values.length - 1];
        cell.innerHTML = `<h4>${label}</h4><p class="scale-delta">${last > 0 ? '+' : ''}${last.toFixed(1)}% at ${xs[xs.length - 1]}</p>`;
        cell.appendChild(svg);
        grid.appendChild(cell);
      });
    }

    document.querySelectorAll('.tab-btn[data-scale]').forEach((tab) => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn[data-scale]').forEach((t) => t.setAttribute('aria-pressed', String(t === tab)));
        render(tab.dataset.scale);
      });
    });

    render('dataset');
  })();

  /* ------------------------------------------------------------------ *
   * Ext. Data Fig. 1 — the GNN–LM wall (scaled MAE, MatBench-style tasks)
   * ------------------------------------------------------------------ */
  (function wall() {
    const panel = document.getElementById('wall-panel');
    if (!panel) return;

    const DATA = {
      'Perovskite FE': [
        { model: 'COGN', type: 'gnn', v: 0.0 }, { model: 'SchNet', type: 'gnn', v: 0.0192 }, { model: 'DimeNet', type: 'gnn', v: 0.0282 },
        { model: 'MODNet', type: 'other', v: 0.1683 },
        { model: 'MatText', type: 'llm', v: 0.1076 }, { model: 'Robocrys', type: 'llm', v: 0.3638 }, { model: 'CrabNet', type: 'llm', v: 1.0 },
      ],
      'Formation Energy': [
        { model: 'COGN', type: 'gnn', v: 0.0 }, { model: 'SchNet', type: 'gnn', v: 0.0304 }, { model: 'DimeNet', type: 'gnn', v: 0.0411 },
        { model: 'MODNet', type: 'other', v: 0.1759 },
        { model: 'CrabNet', type: 'llm', v: 0.438 }, { model: 'MatText', type: 'llm', v: 1.0 },
      ],
      'Bulk modulus': [
        { model: 'COGN', type: 'gnn', v: 0.0 }, { model: 'DimeNet', type: 'gnn', v: 0.0692 }, { model: 'SchNet', type: 'gnn', v: 0.1028 },
        { model: 'MODNet', type: 'other', v: 0.0243 },
        { model: 'CrabNet', type: 'llm', v: 0.4168 }, { model: 'MatText', type: 'llm', v: 0.6075 }, { model: 'Robocrys', type: 'llm', v: 1.0 },
      ],
      'Shear modulus': [
        { model: 'COGN', type: 'gnn', v: 0.0 }, { model: 'DimeNet', type: 'gnn', v: 0.1632 }, { model: 'SchNet', type: 'gnn', v: 0.1696 },
        { model: 'MODNet', type: 'other', v: 0.0666 },
        { model: 'MatText', type: 'llm', v: 0.4805 }, { model: 'CrabNet', type: 'llm', v: 0.5151 }, { model: 'Robocrys', type: 'llm', v: 1.0 },
      ],
      'Band gap': [
        { model: 'COGN', type: 'gnn', v: 0.0 }, { model: 'DimeNet', type: 'gnn', v: 0.1643 }, { model: 'SchNet', type: 'gnn', v: 0.3003 },
        { model: 'MODNet', type: 'other', v: 0.2423 },
        { model: 'Robocrys', type: 'llm', v: 0.2465 }, { model: 'LLM-Prop', type: 'llm', v: 0.3222 }, { model: 'CrabNet', type: 'llm', v: 0.415 }, { model: 'MatText', type: 'llm', v: 1.0 },
      ],
      'Refractive Index': [
        { model: 'COGN', type: 'gnn', v: 0.2682 }, { model: 'SchNet', type: 'gnn', v: 0.4026 },
        { model: 'MODNet', type: 'other', v: 0.0 },
        { model: 'CrabNet', type: 'llm', v: 0.372 }, { model: 'MatText', type: 'llm', v: 1.0 },
      ],
    };

    panel.innerHTML = '';
    Object.entries(DATA).forEach(([prop, models]) => {
      const row = document.createElement('div');
      row.className = 'wall-row';
      const track = document.createElement('div');
      track.className = 'wall-track';

      models.forEach((m) => {
        const dot = document.createElement('button');
        dot.type = 'button';
        dot.className = `wall-dot ${m.type}`;
        dot.style.left = `${m.v * 100}%`;
        dot.setAttribute('aria-label', `${m.model}: scaled MAE ${m.v.toFixed(2)}`);
        dot.innerHTML = `<span class="wall-tip">${m.model} · ${m.v.toFixed(2)}</span>`;
        track.appendChild(dot);
      });

      row.innerHTML = `<div class="wall-row-label">${prop}</div>`;
      row.appendChild(track);
      panel.appendChild(row);
    });
  })();
})();
