/**
 * Flappy Bird AI 競賽平台 —— 本地伺服器（純 Node 內建模組，無外部依賴）
 *
 * 職責：
 *   - 接收學生/工具訓練好的神經網絡模型（JSON），用固定評分賽道算分
 *   - 維護本地排行榜 data/leaderboard.json
 *   - 提供靜態前端（排行榜 + 即時畫面）+ API
 *
 * API 邏輯與 Vercel serverless 共用 src/httpCore.js，確保本地與線上行為一致。
 * 端點：
 *   POST /api/submit        body: { name, model }  -> { id, score, rank, total }
 *   GET  /api/leaderboard   -> 排行榜陣列（含 model，便於前端重播）
 *   GET  /api/model/:id     -> 單筆模型 JSON
 *   GET  /api/benchmark/:id -> 多種子穩健度 benchmark（可下載 JSON）
 */
const http = require('http');
const fs = require('fs');
const path = require('path');
const { handleApi } = require('./src/httpCore.js');

const PORT = process.env.PORT || 3000;
const ROOT = __dirname;
const PUBLIC_DIR = path.join(ROOT, 'public');
const SRC_DIR = path.join(PUBLIC_DIR, 'src');    // 瀏覽器用的遊戲邏輯
const DOCS_DIR = path.join(PUBLIC_DIR, 'docs');   // API.md

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

// 包一層防崩保險：handler 任何未預期錯誤都回 500，而非讓整個程序退出
const server = http.createServer((req, res) => {
  handleApi(req, res)
    .then(handled => { if (!handled) serveStatic(req, res); })
    .catch(e => {
      try { res.writeHead(500, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ error: String(e) })); }
      catch (_) { /* 回應可能已送出，忽略 */ }
    });
});

server.listen(PORT, () => {
  console.log(`Flappy Bird AI 競賽平台已啟動： http://localhost:${PORT}`);
});
