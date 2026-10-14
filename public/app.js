/* 前端邏輯：排行榜 / 競技 / 在線訓練 / API 文檔 四個模組 */
(function () {
  'use strict';
  const C = window.FlappyConfig;
  const { WIDTH, HEIGHT, BIRD_X, BIRD_SIZE, GRAVITY, FLAP_VY, EVAL_SEED, MAX_FRAMES } = C;

  let board = [];
  let rafId = null;
  let speed = 1;                       // 播放倍速（每 frame 多跑幾個 sim step）
  const selected = new Set();          // 勾選要一起競賽的模型 id
  let lastBench = null;                // 最近一次 benchmark 結果（供下載）

  const COLORS = ['#ffd33d', '#ff6b6b', '#4dabf7', '#69db7c', '#da77f2',
    '#ff922b', '#63e6be', '#ffd43b', '#f783ac', '#a5d8ff'];

  const canvas = document.getElementById('game');
  const ctx = canvas.getContext('2d');
  const statusEl = document.getElementById('status');
  const raceInfoEl = document.getElementById('raceInfo');

  /* ==================== 通用工具 ==================== */
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
  }
  function setStatus(t) { statusEl.textContent = t; }
  function cancelRaf() { if (rafId) { cancelAnimationFrame(rafId); rafId = null; } }
  function clampInt(v, lo, hi, dflt) {
    let n = parseInt(v, 10);
    if (!Number.isFinite(n)) n = dflt;
    return Math.max(lo, Math.min(hi, n));
  }
  function clampNum(v, lo, hi, dflt) {
    let n = parseFloat(v);
    if (!Number.isFinite(n)) n = dflt;
    return Math.max(lo, Math.min(hi, n));
  }
  async function copyText(text) {
    try { await navigator.clipboard.writeText(text); return true; }
    catch (e) {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      let ok = false;
      try { ok = document.execCommand('copy'); } catch (_) { ok = false; }
      document.body.removeChild(ta);
      return ok;
    }
  }

  /* ==================== 菜單切頁 ==================== */
  function switchView(name) {
    document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b.dataset.view === name));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    const target = document.getElementById('view-' + name);
    if (target) target.classList.add('active');
    if (name === 'docs') loadDocs();
  }
  document.querySelectorAll('.tab').forEach(btn => {
    btn.onclick = () => switchView(btn.dataset.view);
  });

  /* ==================== 給 Agent 的說明（一鍵複製） ==================== */
  const AGENT_DOC = [
    '你是參與「Flappy Bird AI 訓練競賽」的 AI 代理。目標：訓練一個神經網絡玩 Flappy Bird，提交到排行榜，並回報分數與排名。',
    '',
    '【模型格式】必須是 2→6→1 前饋神經網絡 JSON：',
    '{"arch":[2,6,1],"weights":[[...],...],"biases":[[...],...]}',
    '- 輸入 x=[dx,dy]：dx=(最近管柱左緣x - 鳥x)/畫布寬；dy=(鳥y - 缺口中心y)/畫布高',
    '- 推論：h_j = tanh(weights[0][j]·x + biases[0][j])；o = sigmoid(weights[1][0]·h + biases[1][0])',
    '- 決策：o > 0.5 → 拍翅膀，否則不動',
    '',
    '【訓練】本地執行（產出 model.json）：',
    'node src/train.js --generations 60 --population 25 --out model.json',
    '（或用任何語言 / PyTorch 產出同格式 JSON）',
    '',
    '【提交】POST http://localhost:3000/api/submit',
    '  Content-Type: application/json',
    '  Body: {"name":"你的名字","model":<model.json 內容>}',
    '  回傳: {"id":..,"score":..,"rank":..,"total":..}',
    '',
    '【其他端點】',
    '  GET /api/leaderboard        排行榜（含 model）',
    '  GET /api/model/:id          單筆模型 JSON',
    '  GET /api/benchmark/:id?count=10   多種子穩健度統計',
    '',
    '【評分】固定種子賽道 EVAL_SEED=12345，分數=通過管柱數；所有人同一賽道，公平可比。',
    '【完成後請回報】模型名稱、分數、排名、是否登上榜首。'
  ].join('\n');

  document.getElementById('agentDoc').textContent = AGENT_DOC;
  document.getElementById('copyAgent').onclick = async () => {
    const ok = await copyText(AGENT_DOC);
    const btn = document.getElementById('copyAgent');
    btn.textContent = ok ? '✅ 已複製' : '⚠️ 複製失敗，請手動選取';
    setTimeout(() => { btn.textContent = '📋 一鍵複製'; }, 1500);
  };

  /* ==================== 輕量 Markdown 渲染器（無外部依賴） ==================== */
  function inlineMd(t) {
    let s = escapeHtml(t);
    s = s.replace(/`([^`]+)`/g, (m, c) => '<code>' + c + '</code>');
    s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    return s;
  }
  function renderTable(rows) {
    const cells = r => r.replace(/^\|/, '').replace(/\|$/, '').split('|').map(c => c.trim());
    const isSep = r => /^\|?[\s:|-]+\|?$/.test(r) && r.indexOf('-') !== -1;
    const header = cells(rows[0]);
    const body = rows.slice(1).filter(r => !isSep(r));
    let out = '<div class="table-wrap"><table><thead><tr>'
      + header.map(c => '<th>' + inlineMd(c) + '</th>').join('')
      + '</tr></thead><tbody>';
    for (const r of body) out += '<tr>' + cells(r).map(c => '<td>' + inlineMd(c) + '</td>').join('') + '</tr>';
    return out + '</tbody></table></div>';
  }
  function renderMarkdown(md) {
    const lines = md.replace(/\r\n/g, '\n').split('\n');
    const out = [];
    let para = [];
    let i = 0;

    function flushPara() {
      if (!para.length) return;
      out.push('<p>' + para.map(inlineMd).join('<br>') + '</p>');
      para = [];
    }

    while (i < lines.length) {
      const line = lines[i];
      const t = line.trim();

      // 程式碼區塊
      if (/^```/.test(t)) {
        flushPara();
        const lang = t.slice(3).trim();
        const buf = [];
        i++;
        while (i < lines.length && !/^```/.test(lines[i].trim())) { buf.push(lines[i]); i++; }
        i++;
        out.push('<pre><code' + (lang ? ' class="lang-' + escapeHtml(lang) + '"' : '') + '>'
          + escapeHtml(buf.join('\n')) + '</code></pre>');
        continue;
      }
      if (t === '') { flushPara(); i++; continue; }
      if (/^(-{3,}|\*{3,}|_{3,})$/.test(t)) { flushPara(); out.push('<hr>'); i++; continue; }

      const h = t.match(/^(#{1,6})\s+(.*)$/);
      if (h) {
        flushPara();
        const lvl = h[1].length;
        out.push('<h' + lvl + '>' + inlineMd(h[2]) + '</h' + lvl + '>');
        i++; continue;
      }
      if (/^>/.test(t)) {
        flushPara();
        const buf = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) {
          buf.push(lines[i].replace(/^\s*>\s?/, ''));
          i++;
        }
        out.push('<blockquote>' + buf.map(inlineMd).join('<br>') + '</blockquote>');
        continue;
      }
      if (/^\|/.test(t)) {
        flushPara();
        const rows = [];
        while (i < lines.length && /^\|/.test(lines[i].trim())) { rows.push(lines[i].trim()); i++; }
        out.push(renderTable(rows));
        continue;
      }
      if (/^[-*+]\s+/.test(t)) {
        flushPara();
        const items = [];
        while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
          items.push('<li>' + inlineMd(lines[i].replace(/^\s*[-*+]\s+/, '')) + '</li>');
          i++;
        }
        out.push('<ul>' + items.join('') + '</ul>');
        continue;
      }
      if (/^\d+\.\s+/.test(t)) {
        flushPara();
        const items = [];
        while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
          items.push('<li>' + inlineMd(lines[i].replace(/^\s*\d+\.\s+/, '')) + '</li>');
          i++;
        }
        out.push('<ol>' + items.join('') + '</ol>');
        continue;
      }
      para.push(t);
      i++;
    }
    flushPara();
    return out.join('\n');
  }

  let docsLoaded = false;
  async function loadDocs() {
    const el = document.getElementById('docsContent');
    if (docsLoaded) return;
    try {
      const res = await fetch('/docs/API.md');
      if (!res.ok) throw new Error('HTTP ' + res.status);
      el.innerHTML = renderMarkdown(await res.text());
      docsLoaded = true;
    } catch (e) {
      el.innerHTML = '<p class="hint">API.md 載入失敗：' + escapeHtml(e.message) + '</p>';
    }
  }
  document.getElementById('copyDocs').onclick = async () => {
    let md = '';
    try { md = await (await fetch('/docs/API.md')).text(); } catch (e) { md = ''; }
    const ok = md ? await copyText(md) : false;
    const btn = document.getElementById('copyDocs');
    btn.textContent = ok ? '✅ 已複製' : '⚠️ 複製失敗';
    setTimeout(() => { btn.textContent = '📋 複製 API.md 原文'; }, 1500);
  };

  /* ==================== 排行榜 ==================== */
  async function loadBoard() {
    const res = await fetch('/api/leaderboard');
    board = await res.json();
    renderBoard();
  }

  function renderBoard() {
    const tb = document.querySelector('#board tbody');
    tb.innerHTML = '';
    if (!board.length) {
      tb.innerHTML = '<tr><td colspan="5">尚無提交，快來當第一個！</td></tr>';
      return;
    }
    board.forEach((e, i) => {
      const tr = document.createElement('tr');
      tr.innerHTML =
        `<td><input type="checkbox" class="sel" data-id="${e.id}"></td>` +
        `<td>${i + 1}</td><td>${escapeHtml(e.name)}</td><td>${e.score}</td>` +
        `<td class="ops">` +
        `<button data-play="${e.id}" title="觀看實戰">▶</button>` +
        `<button data-bench="${e.id}" title="Benchmark">📊</button>` +
        `</td>`;
      const chk = tr.querySelector('.sel');
      chk.checked = selected.has(e.id);
      chk.onchange = (ev) => {
        if (ev.target.checked) selected.add(e.id); else selected.delete(e.id);
        updateSelCount();
      };
      tr.querySelector('[data-play]').onclick = () => watchModel(e.model, e.name);
      tr.querySelector('[data-bench]').onclick = () => runBenchmark(e.id, e.name);
      tb.appendChild(tr);
    });
    updateSelCount();
  }

  function updateSelCount() {
    const el = document.getElementById('selCount');
    if (el) el.textContent = selected.size ? `（${selected.size}）` : '';
  }

  document.getElementById('selectAll').onclick = () => {
    board.forEach(e => selected.add(e.id));
    renderBoard();
  };
  document.getElementById('clearSel').onclick = () => {
    selected.clear();
    renderBoard();
  };
  document.getElementById('raceBtn').onclick = () => {
    const entries = board.filter(e => selected.has(e.id));
    if (!entries.length) { setStatus('請先勾選至少一筆要一起競賽的模型。'); return; }
    watchRace(entries);
  };

  /* ==================== 繪圖（單隻 / 多隻共用） ==================== */
  function drawPipes(c, sim) {
    c.fillStyle = '#5a8f3a';
    for (const p of sim.pipes) {
      const gapTop = p.gapY - C.PIPE_GAP / 2;
      const gapBottom = p.gapY + C.PIPE_GAP / 2;
      c.fillRect(p.x, 0, C.PIPE_WIDTH, gapTop);
      c.fillRect(p.x, gapBottom, C.PIPE_WIDTH, HEIGHT - 20 - gapBottom);
    }
  }
  function drawBackground(c) {
    c.fillStyle = '#70c5ce';
    c.fillRect(0, 0, WIDTH, HEIGHT);
    c.fillStyle = '#ded895';
    c.fillRect(0, HEIGHT - 20, WIDTH, 20);
  }
  function drawBird(c, b) {
    c.globalAlpha = b.alive === false ? 0.35 : 1;
    c.fillStyle = b.color || '#ffd33d';
    c.beginPath();
    c.arc(BIRD_X, b.y, BIRD_SIZE / 2, 0, Math.PI * 2);
    c.fill();
    c.lineWidth = 2; c.strokeStyle = '#000'; c.stroke();
    if (b.alive === false) {
      c.beginPath();
      c.moveTo(BIRD_X - 7, b.y - 7); c.lineTo(BIRD_X + 7, b.y + 7);
      c.moveTo(BIRD_X + 7, b.y - 7); c.lineTo(BIRD_X - 7, b.y + 7);
      c.stroke();
    }
    c.globalAlpha = 1;
  }

  function drawScene(sim, birds) {
    drawBackground(ctx);
    drawPipes(ctx, sim);
    for (const b of birds) drawBird(ctx, b);
    ctx.fillStyle = '#fff';
    ctx.font = '20px sans-serif';
    ctx.fillText('Score: ' + sim.score, 10, 28);
  }

  function renderRaceInfo(birds) {
    const sorted = [...birds].sort((a, b) => b.score - a.score);
    raceInfoEl.innerHTML = sorted.map((b, i) => {
      const tag = b.alive ? '🟢' : '💀';
      return `<div><span style="color:${b.color}">■</span> ${i + 1}. ${escapeHtml(b.name)} — ${b.score} 分 ${tag}</div>`;
    }).join('');
  }

  /* ==================== 單隻播放（支援加速） ==================== */
  function watchModel(model, name) {
    cancelRaf();
    const b = {
      net: window.NeuralNet.deserialize(model),
      name: name || '模型', color: '#ffd33d',
      y: HEIGHT / 2, vy: 0, alive: true, score: 0
    };
    const sim = new window.FlappySim(EVAL_SEED);
    raceInfoEl.innerHTML = '';
    setStatus('觀看中：' + b.name + '（評分賽道，分數應與提交時一致）');

    function step() {
      for (let s = 0; s < speed; s++) {
        const inputs = sim.getInputs(b.y);
        if (b.net.activate(inputs)[0] > 0.5) b.vy = FLAP_VY; else b.vy += GRAVITY;
        b.y += b.vy;
        if (sim.collisionAt(b.y)) { b.alive = false; break; }
        const prev = sim.score;
        sim.advance();
        b.score += sim.score - prev;
      }
      drawScene(sim, [b]);
      if (!b.alive) {
        setStatus('💀 ' + b.name + ' 撞柱！最終分數：' + sim.score);
        return;
      }
      rafId = requestAnimationFrame(step);
    }
    step();
  }

  /* ==================== 多 Agent 同場競賽（支援加速） ==================== */
  function watchRace(entries) {
    cancelRaf();
    const birds = entries.map((e, i) => ({
      net: window.NeuralNet.deserialize(e.model),
      name: e.name,
      color: COLORS[i % COLORS.length],
      y: HEIGHT / 2, vy: 0, alive: true, score: 0
    }));
    const sim = new window.FlappySim(EVAL_SEED);
    setStatus('🏁 競賽開始：' + birds.length + ' 隻 Agent 在同一評分賽道競技（倍速 ' + speed + 'x）');
    renderRaceInfo(birds);

    function step() {
      for (let s = 0; s < speed; s++) {
        let anyAlive = false;
        for (const b of birds) {
          if (!b.alive) continue;
          const inputs = sim.getInputs(b.y);
          if (b.net.activate(inputs)[0] > 0.5) b.vy = FLAP_VY; else b.vy += GRAVITY;
          b.y += b.vy;
          if (sim.collisionAt(b.y)) { b.alive = false; continue; }
          anyAlive = true;
        }
        const prev = sim.score;
        sim.advance();
        const delta = sim.score - prev;
        for (const b of birds) if (b.alive) b.score += delta;
        if (!anyAlive) break;
      }
      drawScene(sim, birds);
      renderRaceInfo(birds);
      if (!birds.some(b => b.alive)) {
        const winner = [...birds].sort((a, b) => b.score - a.score)[0];
        setStatus('🏆 競賽結束！冠軍：' + winner.name + '（' + winner.score + ' 分）');
        return;
      }
      rafId = requestAnimationFrame(step);
    }
    step();
  }

  /* ==================== 倍速控制 ==================== */
  document.querySelectorAll('.spd').forEach(btn => {
    btn.onclick = () => {
      speed = parseInt(btn.dataset.speed, 10) || 1;
      document.querySelectorAll('.spd').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (statusEl.textContent.startsWith('🏁')) {
        setStatus('🏁 競賽進行中（倍速 ' + speed + 'x）— 下一幀起生效');
      }
    };
  });

  /* ==================== benchmark ==================== */
  async function runBenchmark(id, name) {
    setStatus('⏳ 正在跑 benchmark（多種子評測）...');
    let data;
    try {
      const res = await fetch('/api/benchmark/' + id + '?count=10');
      data = await res.json();
      if (!res.ok) throw new Error(data.error || 'benchmark 失敗');
    } catch (e) {
      setStatus('benchmark 失敗：' + e.message);
      return;
    }
    lastBench = data;
    showBench(data);
    setStatus('✅ ' + name + ' 的 benchmark 完成：平均 ' + data.summary.avg + ' 分（官方賽道 ' + data.results[0].score + ' 分）');
  }

  function showBench(d) {
    document.getElementById('benchTitle').textContent = '📊 Benchmark：' + d.name + '（' + d.seedsCount + ' 個種子）';
    const rows = d.results.map(r =>
      `<tr><td>${r.seed}</td><td>${r.score}</td><td>${r.frames}</td><td>${r.official ? '⭐ 官方' : ''}</td></tr>`
    ).join('');
    document.getElementById('benchContent').innerHTML =
      '<table class="bench"><thead><tr><th>種子</th><th>分數</th><th>存活幀</th><th></th></tr></thead>' +
      '<tbody>' + rows + '</tbody></table>' +
      '<div class="summary">' +
      '<div>平均：<b>' + d.summary.avg + '</b> 分</div>' +
      '<div>最佳：<b>' + d.summary.best + '</b> 分｜最差：<b>' + d.summary.worst + '</b> 分｜中位：<b>' + d.summary.median + '</b></div>' +
      '<div>滿分存活：' + d.summary.perfectRuns + '/' + d.results.length + '（存活率 ' + d.summary.survivalRate + '%）</div>' +
      '</div>';
    document.getElementById('benchModal').classList.remove('hidden');
  }

  document.getElementById('benchClose').onclick = () => {
    document.getElementById('benchModal').classList.add('hidden');
  };
  document.getElementById('benchDownload').onclick = () => {
    if (!lastBench) return;
    const blob = new Blob([JSON.stringify(lastBench, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'benchmark-' + lastBench.name + '-' + lastBench.id + '.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
  };
  document.getElementById('benchModal').onclick = (ev) => {
    if (ev.target.id === 'benchModal') ev.target.classList.add('hidden');
  };

  /* ==================== 在線訓練 ==================== */
  const trainCanvas = document.getElementById('trainCanvas');
  const tctx = trainCanvas.getContext('2d');
  const chartCanvas = document.getElementById('trainChart');
  const cctx = chartCanvas.getContext('2d');
  let trainState = null;

  function readTrainParams() {
    const p = clampInt(document.getElementById('tPopulation').value, 2, 80, 20);
    let k = clampInt(document.getElementById('tTopK').value, 1, 20, 4);
    if (k > p) k = p;
    const seed = parseInt(document.getElementById('tSeed').value, 10);
    return {
      generations: clampInt(document.getElementById('tGenerations').value, 1, 500, 30),
      population: p,
      topK: k,
      mutationRate: clampNum(document.getElementById('tMutation').value, 0, 1, 0.1),
      seed: Number.isFinite(seed) ? seed : 12345,
      maxFrames: clampInt(document.getElementById('tMaxFrames').value, 200, 20000, 2000),
      speed: clampInt(document.getElementById('tSpeed').value, 1, 60, 15)
    };
  }

  function setTrainProgress(t) { document.getElementById('trainProgress').textContent = t; }

  function startTraining() {
    stopTraining();
    const params = readTrainParams();
    const trainer = new window.FlappyTrainer.Trainer({
      generations: params.generations,
      population: params.population,
      topK: params.topK,
      mutationRate: params.mutationRate,
      seed: params.seed
    });
    trainState = {
      trainer, params, running: true, raf: null,
      sim: new window.FlappySim(EVAL_SEED),
      birds: trainer.pop.map(net => ({ net, y: HEIGHT / 2, vy: 0, alive: true, fit: 0 })),
      frame: 0, lastInfo: null, model: null
    };
    document.getElementById('trainResult').classList.add('hidden');
    document.getElementById('trainLog').innerHTML = '';
    document.getElementById('trainSubmitResult').textContent = '';
    document.getElementById('trainStop').disabled = false;
    document.getElementById('trainStart').disabled = true;
    setTrainProgress('訓練中…（切到其他分頁也會繼續跑）');
    trainLoop();
  }

  function stopTraining() {
    if (trainState && trainState.raf) cancelAnimationFrame(trainState.raf);
    if (trainState) trainState.running = false;
    document.getElementById('trainStop').disabled = true;
    document.getElementById('trainStart').disabled = false;
  }

  function trainLoop() {
    const st = trainState;
    if (!st || !st.running) return;

    let genEnded = false;
    for (let s = 0; s < st.params.speed; s++) {
      let anyAlive = false;
      for (const b of st.birds) {
        if (!b.alive) continue;
        const inputs = st.sim.getInputs(b.y);
        if (b.net.activate(inputs)[0] > 0.5) b.vy = FLAP_VY; else b.vy += GRAVITY;
        b.y += b.vy;
        if (st.sim.collisionAt(b.y)) b.alive = false;
        else b.fit += window.FlappyTrainer.fitnessDelta(inputs);
      }
      st.sim.advance();
      st.frame++;
      anyAlive = st.birds.some(b => b.alive);
      if (!anyAlive || st.frame >= st.params.maxFrames) { genEnded = true; break; }
    }
    drawTraining(st);

    if (genEnded) {
      const info = st.trainer.nextGeneration(st.birds.map(b => b.fit));
      st.lastInfo = info;
      logTrainGen(info);
      drawChart(st.trainer.history);
      setTrainProgress(
        `第 ${info.gen}/${info.total} 代｜本代最佳 ${Math.round(info.best)}｜` +
        `歷史最佳 ${Math.round(info.bestFitness)}｜平均 ${Math.round(info.avg)}` +
        (info.earlyStop ? '｜已達標提前結束' : '')
      );
      if (info.done) { finishTraining(); return; }
      st.sim = new window.FlappySim(EVAL_SEED);
      st.birds = st.trainer.pop.map(net => ({ net, y: HEIGHT / 2, vy: 0, alive: true, fit: 0 }));
      st.frame = 0;
    }
    st.raf = requestAnimationFrame(trainLoop);
  }

  function drawTraining(st) {
    drawBackground(tctx);
    drawPipes(tctx, st.sim);
    let alive = 0;
    for (const b of st.birds) {
      if (!b.alive) continue;
      alive++;
      tctx.fillStyle = '#ffd33d';
      tctx.beginPath();
      tctx.arc(BIRD_X, b.y, BIRD_SIZE / 2 - 1, 0, Math.PI * 2);
      tctx.fill();
      tctx.lineWidth = 1; tctx.strokeStyle = 'rgba(0,0,0,.35)'; tctx.stroke();
    }
    const info = st.lastInfo;
    tctx.fillStyle = '#fff';
    tctx.font = '16px sans-serif';
    const label = info
      ? `Gen ${info.gen}/${info.total}  存活 ${alive}/${st.birds.length}  最佳 ${Math.round(info.bestFitness)}`
      : `第 1 代  存活 ${alive}/${st.birds.length}`;
    tctx.fillText(label, 8, 24);
  }

  function drawChart(history) {
    const w = chartCanvas.width, h = chartCanvas.height;
    cctx.clearRect(0, 0, w, h);
    cctx.fillStyle = '#f8fafc';
    cctx.fillRect(0, 0, w, h);
    if (!history.length) return;
    const maxV = Math.max.apply(null, history.map(x => x.best).concat([1]));
    const pad = 10;
    const iw = w - pad * 2, ih = h - pad * 2;
    cctx.strokeStyle = '#e2e8f0';
    cctx.lineWidth = 1;
    for (let g = 1; g <= 3; g++) {
      const y = pad + ih * (g / 4);
      cctx.beginPath(); cctx.moveTo(pad, y); cctx.lineTo(w - pad, y); cctx.stroke();
    }
    cctx.strokeStyle = '#2563eb';
    cctx.lineWidth = 2;
    cctx.beginPath();
    history.forEach((x, i) => {
      const px = pad + (history.length === 1 ? iw / 2 : (i / (history.length - 1)) * iw);
      const py = pad + ih - (x.best / maxV) * ih;
      if (i === 0) cctx.moveTo(px, py); else cctx.lineTo(px, py);
    });
    cctx.stroke();
    cctx.fillStyle = '#64748b';
    cctx.font = '11px sans-serif';
    cctx.fillText('max ' + Math.round(maxV), pad + 2, pad + 10);
    cctx.fillText('gen ' + history.length, w - 48, h - 4);
  }

  function logTrainGen(info) {
    const log = document.getElementById('trainLog');
    const div = document.createElement('div');
    div.className = 'log-line' + (info.best >= info.bestFitness ? ' log-best' : '');
    div.textContent =
      `Gen ${String(info.gen).padStart(3)}/${info.total} | best=${Math.round(info.best)} ` +
      `avg=${Math.round(info.avg)} topK=[${info.topScores.map(s => Math.round(s)).join(',')}]`;
    log.insertBefore(div, log.firstChild);
    while (log.children.length > 40) log.removeChild(log.lastChild);
  }

  // 用與伺服器相同的邏輯（EVAL_SEED）算真實分數
  function scoreModel(model) {
    const net = window.NeuralNet.deserialize(model);
    const sim = new window.FlappySim(EVAL_SEED);
    let y = HEIGHT / 2, vy = 0, frame = 0;
    while (frame < MAX_FRAMES) {
      const inputs = sim.getInputs(y);
      if (net.activate(inputs)[0] > 0.5) vy = FLAP_VY; else vy += GRAVITY;
      y += vy;
      if (sim.collisionAt(y)) break;
      sim.advance();
      frame++;
    }
    return sim.score;
  }

  function finishTraining() {
    const st = trainState;
    st.running = false;
    document.getElementById('trainStop').disabled = true;
    document.getElementById('trainStart').disabled = false;

    const model = st.trainer.model();
    st.model = model;
    let score = -1;
    try { score = scoreModel(model); } catch (e) { score = -1; }

    document.getElementById('trainScore').innerHTML =
      '評分賽道（EVAL_SEED=' + EVAL_SEED + '）實際分數：<b>' + score + '</b> 分' +
      '（共演化 ' + st.trainer.gen + ' 代，歷史最佳 fitness ' + Math.round(st.trainer.bestFitness) + '）';
    document.getElementById('trainResult').classList.remove('hidden');
    setTrainProgress('✅ 訓練完成！可提交到排行榜、下載模型，或切到「排行榜 / 競技」觀看。');
  }

  document.getElementById('trainStart').onclick = startTraining;
  document.getElementById('trainStop').onclick = stopTraining;
  // 演化有隨機性，分數不理想時可原參數重跑一次
  document.getElementById('trainAgain').onclick = () => startTraining();

  document.getElementById('trainDownload').onclick = () => {
    if (!trainState || !trainState.model) return;
    const blob = new Blob([JSON.stringify(trainState.model, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'model.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
  };
  document.getElementById('trainSubmit').onclick = async () => {
    if (!trainState || !trainState.model) return;
    const name = document.getElementById('trainName').value.trim() || 'Web-Trainer';
    const res = await fetch('/api/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, model: trainState.model })
    });
    const data = await res.json();
    document.getElementById('trainSubmitResult').textContent = JSON.stringify(data, null, 2);
    loadBoard();
    switchView('play');
    setStatus('已送出 ' + name + '：得分 ' + data.score + '，排名 ' + data.rank + '/' + data.total);
  };
  document.getElementById('trainWatch').onclick = () => {
    if (!trainState || !trainState.model) return;
    const name = document.getElementById('trainName').value.trim() || 'Web-Trainer';
    switchView('play');
    watchModel(trainState.model, name);
  };

  /* ==================== 其餘按鈕 / 表單 ==================== */
  document.getElementById('watchBest').onclick = () => {
    if (board[0]) watchModel(board[0].model, board[0].name);
  };
  document.getElementById('watchRandom').onclick = () => {
    if (board.length) {
      const e = board[Math.floor(Math.random() * board.length)];
      watchModel(e.model, e.name);
    }
  };
  document.getElementById('refresh').onclick = loadBoard;

  document.getElementById('submitForm').onsubmit = async (ev) => {
    ev.preventDefault();
    const name = document.getElementById('nameInput').value.trim();
    let modelText = document.getElementById('modelInput').value.trim();
    const fileInput = document.getElementById('modelFile');
    if (fileInput.files.length) modelText = await fileInput.files[0].text();
    let model;
    try { model = JSON.parse(modelText); }
    catch (e) { document.getElementById('submitResult').textContent = 'JSON 解析失敗：' + e; return; }
    const res = await fetch('/api/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, model })
    });
    const data = await res.json();
    document.getElementById('submitResult').textContent = JSON.stringify(data, null, 2);
    loadBoard();
  };

  loadBoard();
})();
