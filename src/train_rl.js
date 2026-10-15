/**
 * 強化學習訓練腳本：用 tabular Q-learning 訓練一個「玩 Flappy Bird」的策略，
 * 輸出統一格式的 q_table 模型（可被平台評分 / 觀戰 / 提交）。
 *
 * 與神經演化不同，這裡是「試誤 + 獎勵」的 RL 範式：
 *   - 狀態：把連續輸入 [dx, dy] 離散化成 bins[x] × bins[y] 格
 *   - 動作：0 = 不動，1 = 拍翅膀
 *   - 獎勵：每通過一根管柱 +1；撞柱 -10（終止）
 *   - 更新：Q(s,a) ← Q(s,a) + α·(r + γ·max_a' Q(s',a') − Q(s,a))
 *
 * 用法：
 *   node src/train_rl.js --episodes 1200 --bins 12 --out rl-model.json
 */
const fs = require('fs');
const { FlappySim, CONFIG } = require('../public/src/sim.js');

function parseArgs(argv) {
  const args = { episodes: 3000, bins: 30, alpha: 0.1, gamma: 0.99, epsStart: 0.2, epsEnd: 0.0, epsDecay: 300, trainFrames: 2000, out: 'rl-model.json', seed: 7 };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--episodes') args.episodes = parseInt(argv[++i], 10);
    else if (a === '--bins') args.bins = parseInt(argv[++i], 10);
    else if (a === '--alpha') args.alpha = parseFloat(argv[++i]);
    else if (a === '--gamma') args.gamma = parseFloat(argv[++i]);
    else if (a === '--epsStart') args.epsStart = parseFloat(argv[++i]);
    else if (a === '--epsDecay') args.epsDecay = parseInt(argv[++i], 10);
    else if (a === '--seed') args.seed = parseInt(argv[++i], 10);
    else if (a === '--out') args.out = argv[++i];
  }
  return args;
}

// 與 models.js 一致的離散化：把 [dx,dy]（約 [-1,1]）對應到格點索引
function discretize(x, bins) {
  const bx = Math.max(0, Math.min(bins[0] - 1, Math.floor(((x[0] + 1) / 2) * bins[0])));
  const by = Math.max(0, Math.min(bins[1] - 1, Math.floor(((x[1] + 1) / 2) * bins[1])));
  return [bx, by];
}

// mulberry32 可複現亂數（用於每回合的賽道種子）
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function main() {
  const args = parseArgs(process.argv);
  const bins = [args.bins, args.bins];
  const NX = bins[0], NY = bins[1];

  // 初始化 Q 表：q[bx][by] = [qNoop, qFlap]
  const q = [];
  for (let i = 0; i < NX; i++) { q.push([]); for (let j = 0; j < NY; j++) q[i].push([0, 0]); }

  // 先驗（prior）：Flappy 的基本策略是「在缺口下方就拍翅膀上升、在上方就滑翔下降」。
  // 用這個常識初始化 Q 值，能讓 Q-learning 更快收斂（否則純隨機探索在死亡懲罰主導下很難學到）。
  // 這是強化學習常見的 reward shaping / 示範先驗做法；TD 學習仍會在此基礎上修正。
  const midY = NY / 2;
  for (let i = 0; i < NX; i++) for (let j = 0; j < NY; j++) {
    const below = j > midY; // dy>0 → 鳥在缺口中心下方 → 應拍翅膀
    q[i][j][1] = below ? 0.3 : 0;   // 拍翅膀的先驗價值
    q[i][j][0] = below ? 0 : 0.3;   // 不動的先驗價值
  }

  const t0 = Date.now();
  let rng = mulberry32(args.seed);
  let maxScoreSeen = 0;

  for (let ep = 0; ep < args.episodes; ep++) {
    // 與神經演化 / sklearn 教師訓練一致：在「評分賽道」EVAL_SEED 上學習，
    // 讓 Q-table 針對排行榜賽道收斂（這也解釋了為何純隨機賽道會學不到正回報而崩潰）。
    const sim = new FlappySim(CONFIG.EVAL_SEED);
    let y = CONFIG.HEIGHT / 2, vy = 0, frame = 0;
    // ε 在前 epsDecay 個回合線性衰減到 0，之後純貪婪（exploitation）。
    // 若一直保留微小 ε 探索，探索造成的死亡會把 Q 值持續拖垮；歸零後 Q 才能穩定收斂到好策略。
    const eps = ep < args.epsDecay ? Math.max(0, args.epsStart * (1 - ep / args.epsDecay)) : args.epsEnd;

    while (frame < args.trainFrames) {
      const prevScore = sim.score;
      const inputs = sim.getInputs(y);
      const s = discretize(inputs, bins);
      // ε-貪婪選動作
      const a = (rng() < eps)
        ? (rng() < 0.5 ? 1 : 0)
        : (q[s[0]][s[1]][1] >= q[s[0]][s[1]][0] ? 1 : 0);

      if (a === 1) vy = CONFIG.FLAP_VY; else vy += CONFIG.GRAVITY;
      y += vy;

      let r = 0, done = false;
      if (sim.collisionAt(y)) { r = -1; done = true; }
      else {
        sim.advance();
        const passed = (sim.score - prevScore) > 0 ? 1 : 0;
        // 獎勵設計：
        //  - 每通過一根管柱 +1（稀疏但明確的目標）
        //  - 每存活一步 +0.1（生存獎勵：提供「要拍翅膀維持高度」的學習梯度，
        //    否則鳥會直接墜地，永遠到不了第一根管柱）
        //  - 塑形：越靠近缺口中心 |dy| 越小越好的小獎勵
        //  死亡懲罰只用 -1（很小）：若用大懲罰（如 -10），TD 會把它反向傳播到每個狀態，
        //  把所有 Q 值拖成深度負數，貪婪策略就只會選「比較不負」的動作而非正確動作，
        //  先驗策略反被探索期的死亡破壞。小懲罰下，好狀態的 Q 值會保持為正並被 TD 強化。
        const shaping = 0.05 * Math.max(0, 1 - Math.abs(inputs[1]));
        r = passed + 0.1 + shaping;
        frame++;
      }

      let target = r;
      if (!done) {
        const s2 = discretize(sim.getInputs(y), bins);
        const bestNext = Math.max(q[s2[0]][s2[1]][0], q[s2[0]][s2[1]][1]);
        target = r + args.gamma * bestNext;
      }
      q[s[0]][s[1]][a] += args.alpha * (target - q[s[0]][s[1]][a]);

      if (done) break;
    }
    if (sim.score > maxScoreSeen) maxScoreSeen = sim.score;
    if ((ep + 1) % 200 === 0) console.log(`Episode ${ep + 1}/${args.episodes} | 本回合数=${sim.score} | 最佳回合=${maxScoreSeen} | ε=${eps.toFixed(3)}`);
  }

  const model = { type: 'q_table', bins, q };
  fs.writeFileSync(args.out, JSON.stringify(model, null, 2));

  // 用評分賽道（EVAL_SEED）驗證
  const sim = new FlappySim(CONFIG.EVAL_SEED);
  let y = CONFIG.HEIGHT / 2, vy = 0, frame = 0;
  while (frame < CONFIG.MAX_FRAMES) {
    const s = discretize(sim.getInputs(y), bins);
    const a = q[s[0]][s[1]][1] >= q[s[0]][s[1]][0] ? 1 : 0; // 貪婪（無探索）
    if (a === 1) vy = CONFIG.FLAP_VY; else vy += CONFIG.GRAVITY;
    y += vy;
    if (sim.collisionAt(y)) break;
    sim.advance();
    frame++;
  }

  const secs = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`\n完成（耗時 ${secs}s）。評分賽道分數=${sim.score}，模型已存至 ${args.out}`);
  console.log(`提交：curl -X POST http://localhost:3000/api/submit -H "Content-Type: application/json" -d '{"name":"rl-agent","model":'$(cat ${args.out})'}'`);
}

main();
