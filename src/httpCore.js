/**
 * 共用核心：同時給本地 server.js 與 Vercel serverless (api/[...slug].js) 使用。
 * 負責：排行榜讀寫、模型評分、多種子 benchmark。
 * 遊戲邏輯來自 public/src/neuralnet.js 與 public/src/sim.js（Node / 瀏覽器同份）。
 */
const fs = require('fs');
const path = require('path');
const { NeuralNet } = require('../public/src/neuralnet.js');
const { FlappySim, CONFIG } = require('../public/src/sim.js');

const ROOT = path.join(__dirname, '..');
const DATA_DIR = path.join(ROOT, 'data');
const DATA_FILE = path.join(DATA_DIR, 'leaderboard.json');

/* 用 require 載入種子排行榜——這樣 Vercel serverless 的依賴追蹤（nft）會把它打進函式包，
 * 冷啟動時也能拿到預先放好的示範資料。 */
let SEED_BOARD = [];
try { SEED_BOARD = require('../data/leaderboard.json'); } catch (e) { SEED_BOARD = []; }

/* ---- 排行榜儲存 ----
 * 本地：直接寫回 data/leaderboard.json。
 * Vercel serverless：運行時檔案系統唯讀，寫入改存 /tmp（同實例內有效；冷啟動後回到種子資料）。
 * 兩者都先用記憶體快取，避免重複讀檔。 */
let boardCache = null;
function loadBoard() {
  if (boardCache) return boardCache;
  let b = null;
  try { b = JSON.parse(fs.readFileSync('/tmp/leaderboard.json', 'utf8')); } catch (e) { /* 沒有快取 */ }
  if (!b) b = SEED_BOARD;
  boardCache = b;
  return b;
}
function saveBoard(b) {
  boardCache = b;
  try { fs.writeFileSync('/tmp/leaderboard.json', JSON.stringify(b, null, 2)); } catch (e) { /* 唯讀環境忽略 */ }
  try { if (fs.existsSync(DATA_DIR)) fs.writeFileSync(DATA_FILE, JSON.stringify(b, null, 2)); } catch (e) { /* 唯讀環境忽略 */ }
}

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

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', c => (body += c));
    req.on('end', () => resolve(body));
    req.on('error', reject);
  });
}

/**
 * 處理所有 /api/* 請求。
 * 回傳 true 表示已處理（已寫入 res）；false 表示不是 api 請求（交由靜態伺服器處理）。
 */
async function handleApi(req, res) {
  const url = req.url.split('?')[0];

  if (req.method === 'POST' && url === '/api/submit') {
    try {
      const data = JSON.parse(await readBody(req));
      if (!data.name || !data.model) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'name 與 model 皆為必填' }));
        return true;
      }
      const model = data.model;
      if (!model.arch || JSON.stringify(model.arch) !== JSON.stringify([2, 6, 1])) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'model.arch 必須為 [2,6,1]' }));
        return true;
      }
      let score;
      try {
        score = evaluateModel(model, CONFIG.EVAL_SEED, CONFIG.MAX_FRAMES);
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: '模型計算失敗，請檢查 weights/biases 格式：' + e.message }));
        return true;
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
    return true;
  }

  if (req.method === 'GET' && url === '/api/leaderboard') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(loadBoard()));
    return true;
  }

  // 多種子 benchmark：測模型穩健度（非只會作弊官方賽道），結果可下載
  if (req.method === 'GET' && url.startsWith('/api/benchmark/')) {
    const id = url.slice('/api/benchmark/'.length);
    const entry = loadBoard().find(x => x.id === id);
    if (!entry) { res.writeHead(404, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ error: 'not found' })); return true; }
    const parsed = new URL(req.url, 'http://localhost');
    let count = parseInt(parsed.searchParams.get('count') || '10', 10);
    if (!Number.isFinite(count) || count < 1) count = 10;
    if (count > 50) count = 50;

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
      const response = {
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
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(response, null, 2));
    } catch (e) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'benchmark 計算失敗，請檢查模型格式：' + e.message }));
    }
    return true;
  }

  if (req.method === 'GET' && url.startsWith('/api/model/')) {
    const id = url.split('/').pop();
    const entry = loadBoard().find(x => x.id === id);
    if (!entry) { res.writeHead(404); res.end('not found'); return true; }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(entry.model));
    return true;
  }

  if (url.startsWith('/api/')) {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'unknown endpoint' }));
    return true;
  }

  return false; // 不是 api 請求
}

module.exports = { handleApi, loadBoard, saveBoard, evaluateModel };
