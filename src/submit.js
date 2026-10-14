/**
 * 提交 CLI：把本地 model.json 送到競賽平台伺服器
 * 用法：node src/submit.js <model.json> <名字> [host] [port]
 *   例：node src/submit.js model.json "學生-A" localhost 3000
 */
const http = require('http');
const fs = require('fs');

const args = process.argv.slice(2);
if (args.length < 2) {
  console.error('用法：node src/submit.js <model.json> <名字> [host] [port]');
  process.exit(1);
}
const file = args[0];
const name = args[1];
const host = args[2] || 'localhost';
const port = args[3] || 3000;

let model;
try {
  model = JSON.parse(fs.readFileSync(file, 'utf8'));
} catch (e) {
  console.error('讀取/解析 model.json 失敗：', e.message);
  process.exit(1);
}

const payload = JSON.stringify({ name, model });
const req = http.request({
  host, port, path: '/api/submit', method: 'POST',
  headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) }
}, res => {
  let data = '';
  res.on('data', c => data += c);
  res.on('end', () => {
    console.log(`HTTP ${res.statusCode}`);
    console.log(data);
  });
});
req.on('error', e => { console.error('送出失敗：', e.message); process.exit(1); });
req.write(payload);
req.end();
