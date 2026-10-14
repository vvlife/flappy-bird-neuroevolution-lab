/**
 * Flappy Bird AI 競賽平台 —— 伺服器（純 Node 內建模組，無外部依賴）
 *
 * 職責：
 *   - 接收學生/工具訓練好的神經網絡模型（JSON），用固定評分賽道算分
 *   - 維護本地排行榜 data/leaderboard.json
 *   - 提供靜態前端（排行榜 + 即時畫面）
 *
 * 端點：
 *   POST /api/submit        body: { name, model }  -> { id, score, rank, total }
 *   GET  /api/leaderboard   -> 排行榜陣列（含 model，便於前端重播）
 *   GET  /api/model/:id     -> 單筆模型 JSON
 *   GET  /api/benchmark/:id -> 多種子穩健度 benchmark（可下載 JSON）
 */
const http = require('http');
const fs = require('fs');
const path = require('path');
const { NeuralNet } = require('./src/neuralnet.js');
const { FlappySim, CONFIG } = require('./src/sim.js');

const PORT = process.env.PORT || 3000;
const ROOT = __dirname;
const DATA_DIR = path.join(ROOT, 'data');
const DATA_FILE = path.join(DATA_DIR, 'leaderboard.json');
const PUBLIC_DIR = path.join(ROOT, 'public');
const SRC_DIR = path.join(ROOT, 'src');
const DOCS_DIR = path.join(ROOT, 'docs');

if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
if (!fs.existsSync(DATA_FILE)) fs.writeFileSync(DATA_FILE, '[]');

function loadBoard() {
  try { return JSON.parse(fs.readFileSync(DATA_FILE, 'utf8')); }
  catch (e) { return []; }
}
function saveBoard(b) { fs.writeFileSync(DATA_FILE, JSON.stringify(b, null, 2)); }

// 用固定評分種子 EVAL_SEED 跑模型，回傳通過的管柱數
function evaluateModel(model, seed, maxFrames) {
  const net = NeuralNet.deserialize(model);
  const sim = new FlappySim(seed);
  let y = CONFIG.HEIGHT / 2;
  let vy = 0;
  let frame = 0;
  while (frame < maxFrames) {
    const inputs = sim.getInputs(y);
    if (net.activate(inputs)[0] > 0.5) vy = CONFIG.FLAP_VY; else vy += CONFIG.GRAVITY;
    y += vy;
    if (sim.collisionAt(y)) break;
    sim.advance();
    frame++;
  }
  return sim.score;
}

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8'
};

function serveStatic(req, res) {
  let urlPath = req.url.split('?')[0];
  if (urlPath === '/') urlPath = '/index.html';
  let filePath;
  if (urlPath.startsWith('/src/')) filePath = path.join(SRC_DIR, urlPath.replace('/src/', ''));
  else if (urlPath.startsWith('/docs/')) filePath = path.join(DOCS_DIR, urlPath.replace('/docs/', ''));
  else filePath = path.join(PUBLIC_DIR, urlPath);

  const resolved = path.resolve(filePath);
  const allowed = [path.resolve(PUBLIC_DIR), path.resolve(SRC_DIR), path.resolve(DOCS_DIR)];
  if (!allowed.some(d => resolved.startsWith(d))) {
    res.writeHead(403); res.end('Forbidden'); return;
  }
  fs.readFile(filePath, (err, data) => {
    if (err) { res.writeHead(404); res.end('Not found'); return; }
    const ext = path.extname(filePath);
    res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream' });
    res.end(data);
  });
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', c => (body += c));
    req.on('end', () => resolve(body));
    req.on('error', reject);
  });
}

async function handle(req, res) {
  const url = req.url.split('?')[0];

  if (req.method === 'POST' && url === '/api/submit') {
    try {
      const data = JSON.parse(await readBody(req));
      if (!data.name || !data.model) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'name 與 model 皆為必填' }));
        return;
      }
      const model = data.model;
      if (!model.arch || JSON.stringify(model.arch) !== JSON.stringify([2, 6, 1])) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'model.arch 必須為 [2,6,1]' }));
        return;
      }
      let score;
      try {
        score = evaluateModel(model, CONFIG.EVAL_SEED, CONFIG.MAX_FRAMES);
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: '模型計算失敗，請檢查 weights/biases 格式：' + e.message }));
        return;
      }
      const board = loadBoard();
      const id = 'm' + Date.now().toString(36) + Math.floor(Math.random() * 1000).toString(36);
      const entry = {
        id,
        name: String(data.name).slice(0, 40),
        score,
        arch: model.arch,
        model,
        createdAt: new Date().toISOString()
      };
      board.push(entry);
      board.sort((a, b) => b.score - a.score);
      saveBoard(board);
      const rank = board.findIndex(e => e.id === id) + 1;
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ id, score, rank, total: board.length }));
    } catch (e) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: String(e) }));
    }
    return;
  }

  if (req.method === 'GET' && url === '/api/leaderboard') {
    const board = loadBoard();
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(board));
    return;
  }

  // 多種子 benchmark：測模型穩健度（非只會作弊官方賽道），結果可下載
  if (req.method === 'GET' && url.startsWith('/api/benchmark/')) {
    const id = url.slice('/api/benchmark/'.length);
    const entry = loadBoard().find(x => x.id === id);
    if (!entry) { res.writeHead(404, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ error: 'not found' })); return; }
    const parsed = new URL(req.url, 'http://localhost');
    let count = parseInt(parsed.searchParams.get('count') || '10', 10);
    if (!Number.isFinite(count) || count < 1) count = 10;
    if (count > 50) count = 50;

    let response;
    try {
      // 第一種子永遠是官方 EVAL_SEED，其餘為固定替代種子（可複現）
      const seeds = [CONFIG.EVAL_SEED];
      for (let i = 1; i < count; i++) seeds.push(1000 * i + 7);

      const results = seeds.map(seed => {
        const net = NeuralNet.deserialize(entry.model);
        const sim = new FlappySim(seed);
        let y = CONFIG.HEIGHT / 2, vy = 0, frame = 0;
        while (frame < CONFIG.MAX_FRAMES) {
          const inputs = sim.getInputs(y);
          if (net.activate(inputs)[0] > 0.5) vy = CONFIG.FLAP_VY; else vy += CONFIG.GRAVITY;
          y += vy;
          if (sim.collisionAt(y)) break;
          sim.advance();
          frame++;
        }
        const perfect = frame >= CONFIG.MAX_FRAMES;
        return { seed, score: sim.score, frames: frame, official: seed === CONFIG.EVAL_SEED, perfect };
      });

      const scores = results.map(r => r.score);
      const sorted = [...scores].sort((a, b) => a - b);
      const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
      const perfectRuns = results.filter(r => r.perfect).length;
      response = {
        id: entry.id,
        name: entry.name,
        arch: entry.arch,
        evalSeed: CONFIG.EVAL_SEED,
        seedsCount: seeds.length,
        results,
        summary: {
          avg: Math.round(avg * 100) / 100,
          best: sorted[sorted.length - 1],
          worst: sorted[0],
          median: sorted[Math.floor(sorted.length / 2)],
          perfectRuns,
          survivalRate: Math.round((perfectRuns / scores.length) * 1000) / 10
        },
        generatedAt: new Date().toISOString()
      };
    } catch (e) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'benchmark 計算失敗，請檢查模型格式：' + e.message }));
      return;
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(response, null, 2));
    return;
  }

  if (req.method === 'GET' && url.startsWith('/api/model/')) {
    const id = url.split('/').pop();
    const entry = loadBoard().find(x => x.id === id);
    if (!entry) { res.writeHead(404); res.end('not found'); return; }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(entry.model));
    return;
  }

  if (url.startsWith('/api/')) {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'unknown endpoint' }));
    return;
  }

  serveStatic(req, res);
}

// 包一層防崩保險：handler 任何未預期錯誤都回 500，而非讓整個程序退出
const server = http.createServer((req, res) => {
  handle(req, res).catch(e => {
    try { res.writeHead(500, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ error: String(e) })); }
    catch (_) { /* 回應可能已送出，忽略 */ }
  });
});

server.listen(PORT, () => {
  console.log(`Flappy Bird AI 競賽平台已啟動： http://localhost:${PORT}`);
});
