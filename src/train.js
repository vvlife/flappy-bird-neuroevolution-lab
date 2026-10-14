/**
 * 訓練腳本：用「神經演化（Neuro-Evolution）」訓練 Flappy Bird 玩家
 *
 * 流程（每一代）：
 *   1. 建立一個種群（population）的隨機神經網絡
 *   2. 讓全體鳥在同一条賽道上同時遊玩（共享管柱序列，公平比較）
 *   3. 依「通過管柱數」計算每隻鳥的 fitness
 *   4. 用遺傳算子產生下一代：菁英保留 + 交叉(crossover) + 突變(mutation)
 *   5. 重複直到世代數用盡，匯出最佳模型 model.json
 *
 * 用法：
 *   node src/train.js --generations 40 --population 20 --topK 4 --mutationRate 0.1 --seed 7 --out model.json
 */
const fs = require('fs');
const { NeuralNet } = require('./neuralnet.js');
const { FlappySim, CONFIG } = require('./sim.js');

function parseArgs(argv) {
  const args = {
    generations: 40,
    population: 20,
    topK: 4,
    mutationRate: 0.1,
    seed: (Date.now() >>> 0) % 2147483647,
    out: 'model.json',
    hidden: 6
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--generations') args.generations = parseInt(argv[++i], 10);
    else if (a === '--population') args.population = parseInt(argv[++i], 10);
    else if (a === '--topK') args.topK = parseInt(argv[++i], 10);
    else if (a === '--mutationRate') args.mutationRate = parseFloat(argv[++i]);
    else if (a === '--seed') args.seed = parseInt(argv[++i], 10);
    else if (a === '--out') args.out = argv[++i];
    else if (a === '--hidden') args.hidden = parseInt(argv[++i], 10);
  }
  return args;
}

// 同一代的所有鳥共用同一條賽道（同 seed）。
// 訓練用的 fitness = 存活 frame 數 + 接近缺口中心的獎勵（即便還沒通過管柱也有梯度，
// 讓演化能從 0 開始往上爬）。排行榜分數則另由伺服器用「通過管柱數」客觀計算。
function evaluatePopulation(nets, seed, maxFrames) {
  const sim = new FlappySim(seed);
  const birds = nets.map(net => ({ net, y: CONFIG.HEIGHT / 2, vy: 0, alive: true, fit: 0 }));
  let frame = 0;
  while (frame < maxFrames && birds.some(b => b.alive)) {
    for (const b of birds) {
      if (!b.alive) continue;
      const inputs = sim.getInputs(b.y);
      const dy = inputs[1];
      const flap = b.net.activate(inputs)[0] > 0.5;
      if (flap) b.vy = CONFIG.FLAP_VY; else b.vy += CONFIG.GRAVITY;
      b.y += b.vy;
      if (sim.collisionAt(b.y)) b.alive = false;
      else b.fit += 1 + Math.max(0, 1 - Math.abs(dy)); // 存活 + 接近缺口中心
    }
    sim.advance();
    frame++;
  }
  return birds.map(b => b.fit);
}

function crossover(a, b) {
  const child = new NeuralNet(a.arch);
  child.weights = [];
  child.biases = [];
  for (let l = 0; l < a.weights.length; l++) {
    const wA = a.weights[l], wB = b.weights[l];
    const cw = wA.map((row, i) => row.map((v, j) => (Math.random() < 0.5 ? v : wB[i][j])));
    child.weights.push(cw);
    child.biases.push(a.biases[l].map((v, i) => (Math.random() < 0.5 ? v : b.biases[l][i])));
  }
  return child;
}

function mutate(net, rate) {
  for (let l = 0; l < net.weights.length; l++) {
    for (let i = 0; i < net.weights[l].length; i++) {
      for (let j = 0; j < net.weights[l][i].length; j++) {
        if (Math.random() < rate) net.weights[l][i][j] += (Math.random() * 2 - 1);
      }
    }
    for (let i = 0; i < net.biases[l].length; i++) {
      if (Math.random() < rate) net.biases[l][i] += (Math.random() * 2 - 1);
    }
  }
  return net;
}

function main() {
  const args = parseArgs(process.argv);
  const arch = [2, args.hidden, 1];
  let population = [];
  for (let i = 0; i < args.population; i++) population.push(NeuralNet.random(arch));

  let best = null;
  let bestScore = -1;
  const t0 = Date.now();

  for (let gen = 0; gen < args.generations; gen++) {
    // 每一代都在「評分賽道」(EVAL_SEED) 上演化，確保訓練出的模型在排行榜上表現好。
    // 若要練「泛化能力」，可改成 args.seed + gen 使用多種子。
    const scores = evaluatePopulation(population, CONFIG.EVAL_SEED, CONFIG.MAX_FRAMES);
    const ranked = population
      .map((net, i) => ({ net, score: scores[i] }))
      .sort((x, y) => y.score - x.score);

    if (ranked[0].score > bestScore) {
      bestScore = ranked[0].score;
      best = ranked[0].net;
    }

    const avg = scores.reduce((s, v) => s + v, 0) / scores.length;
    const top = ranked.slice(0, args.topK);
    console.log(
      `Gen ${String(gen + 1).padStart(3)}/${args.generations} | best=${ranked[0].score} avg=${avg.toFixed(1)} ` +
      `topK=[${top.map(t => t.score).join(',')}]`
    );

    // 早停：模型已能穩定存活很久（約可通過多根管柱）就提早結束
    if (bestScore >= 3000) { console.log('已能穩定存活（fit>=3000），提前結束訓練。'); break; }

    const next = top.map(t => t.net); // 菁英直接保留
    while (next.length < args.population) {
      const p1 = top[Math.floor(Math.random() * top.length)].net;
      const p2 = top[Math.floor(Math.random() * top.length)].net;
      const child = crossover(p1, p2);
      mutate(child, args.mutationRate);
      next.push(child);
    }
    population = next;
  }

  const model = best.serialize();
  fs.writeFileSync(args.out, JSON.stringify(model, null, 2));
  const secs = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`\n完成（耗時 ${secs}s）。最佳分數=${bestScore}，模型已存至 ${args.out}`);
  console.log(`提交：curl -X POST http://localhost:3000/api/submit -H "Content-Type: application/json" ` +
    `-d '{"name":"your-name","model":'$(cat ${args.out})'}'`);
}

main();
