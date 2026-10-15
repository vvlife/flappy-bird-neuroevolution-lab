/**
 * 機器學習風格訓練腳本：用「決策樹 / 隨機森林」訓練一個玩 Flappy Bird 的策略，
 * 輸出統一格式的 decision_tree / random_forest 模型（可被平台評分 / 觀戰 / 提交）。
 *
 * 這是一條「模仿學習（imitation learning）」管線，對照 sklearn 的標準流程：
 *   1. 先用本平台的神經演化（GA）快速訓練一個「老師」神經網絡（夠強即可）；
 *   2. 讓老師在許多賽道裡飛，蒐集 (輸入 [dx,dy], 老師動作 +1/-1) 的訓練資料；
 *   3. 用 CART 演算法從這批資料擬合一棵「回歸決策樹」（葉子輸出 flap 的傾向）；
 *   4. 隨機森林 = 對多個 bootstrap 子樣本、隨機選特徵擬合多棵樹再平均。
 *
 * 這證明了：平台不只收神經網絡，也能收 sklearn 風格的「可解釋」模型。
 *
 * 用法：
 *   node src/train_sklearn.js --mode tree   --out tree-model.json
 *   node src/train_sklearn.js --mode forest --out rf-model.json
 */
const fs = require('fs');
const { NeuralNet } = require('../public/src/neuralnet.js');
const { FlappySim, CONFIG } = require('../public/src/sim.js');

function parseArgs(argv) {
  const args = {
    mode: 'tree', out: 'sklearn-model.json',
    teacherGens: 10, teacherPop: 30, teacherTopK: 4, teacherMut: 0.1, teacherSeed: 11,
    episodes: 25, maxSteps: 1000, seed: 123,
    maxDepth: 5, minSamplesSplit: 8, nTrees: 10
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--mode') args.mode = argv[++i];
    else if (a === '--out') args.out = argv[++i];
    else if (a === '--teacher-gens') args.teacherGens = parseInt(argv[++i], 10);
    else if (a === '--episodes') args.episodes = parseInt(argv[++i], 10);
    else if (a === '--max-depth') args.maxDepth = parseInt(argv[++i], 10);
    else if (a === '--min-split') args.minSamplesSplit = parseInt(argv[++i], 10);
    else if (a === '--n-trees') args.nTrees = parseInt(argv[++i], 10);
    else if (a === '--seed') args.seed = parseInt(argv[++i], 10);
  }
  return args;
}

/* ---------- 1. 神經演化老師（精簡版 GA，與 src/train.js 同一套邏輯） ---------- */
function evaluatePopulation(nets, seed, maxFrames) {
  const sim = new FlappySim(seed);
  const birds = nets.map(net => ({ net, y: CONFIG.HEIGHT / 2, vy: 0, alive: true, fit: 0 }));
  let frame = 0;
  while (frame < maxFrames && birds.some(b => b.alive)) {
    for (const b of birds) {
      if (!b.alive) continue;
      const inputs = sim.getInputs(b.y);
      const dy = inputs[1];
      if (b.net.activate(inputs)[0] > 0.5) b.vy = CONFIG.FLAP_VY; else b.vy += CONFIG.GRAVITY;
      b.y += b.vy;
      if (sim.collisionAt(b.y)) b.alive = false;
      else b.fit += 1 + Math.max(0, 1 - Math.abs(dy));
    }
    sim.advance();
    frame++;
  }
  return birds.map(b => b.fit);
}
function crossover(a, b) {
  const child = new NeuralNet(a.arch);
  child.weights = []; child.biases = [];
  for (let l = 0; l < a.weights.length; l++) {
    child.weights.push(a.weights[l].map((row, i) => row.map((v, j) => (Math.random() < 0.5 ? v : b.weights[l][i][j]))));
    child.biases.push(a.biases[l].map((v, i) => (Math.random() < 0.5 ? v : b.biases[l][i])));
  }
  return child;
}
function mutate(net, rate) {
  for (let l = 0; l < net.weights.length; l++) {
    for (let i = 0; i < net.weights[l].length; i++)
      for (let j = 0; j < net.weights[l][i].length; j++)
        if (Math.random() < rate) net.weights[l][i][j] += (Math.random() * 2 - 1);
    for (let i = 0; i < net.biases[l].length; i++)
      if (Math.random() < rate) net.biases[l][i] += (Math.random() * 2 - 1);
  }
  return net;
}
function trainTeacher(args) {
  const arch = [2, 6, 1];
  let pop = [];
  for (let i = 0; i < args.teacherPop; i++) pop.push(NeuralNet.random(arch));
  let best = null, bestScore = -1;
  for (let gen = 0; gen < args.teacherGens; gen++) {
    // 老師只需「夠強」即可（不追求滿分），用較短的評分上限加速訓練
    const scores = evaluatePopulation(pop, CONFIG.EVAL_SEED, 2000);
    const ranked = pop.map((net, i) => ({ net, score: scores[i] })).sort((x, y) => y.score - x.score);
    if (ranked[0].score > bestScore) { bestScore = ranked[0].score; best = ranked[0].net; }
    if (bestScore >= 3000) break;
    const top = ranked.slice(0, args.teacherTopK);
    const next = top.map(t => t.net);
    while (next.length < args.teacherPop) {
      const p1 = top[Math.floor(Math.random() * top.length)].net;
      const p2 = top[Math.floor(Math.random() * top.length)].net;
      next.push(mutate(crossover(p1, p2), args.teacherMut));
    }
    pop = next;
  }
  console.log(`  老師訓練完成：最佳 fitness=${Math.round(bestScore)}`);
  return best;
}

/* ---------- 2. 蒐集 (輸入, 老師動作) 訓練資料 ---------- */
function collectDataset(teacher, args) {
  const X = [], Y = [];
  for (let ep = 0; ep < args.episodes; ep++) {
    const sim = new FlappySim(args.seed + ep * 7919);
    let y = CONFIG.HEIGHT / 2, vy = 0, frame = 0;
    while (frame < args.maxSteps) {
      const inputs = sim.getInputs(y);
      const flap = teacher.activate(inputs)[0] > 0.5;
      X.push([inputs[0], inputs[1]]);
      Y.push(flap ? 1 : -1);
      if (flap) vy = CONFIG.FLAP_VY; else vy += CONFIG.GRAVITY;
      y += vy;
      if (sim.collisionAt(y)) break;
      sim.advance();
      frame++;
    }
  }
  console.log(`  蒐集訓練資料：${X.length} 筆（動作=拍翅膀佔 ${Math.round(100 * Y.filter(v => v > 0).length / Y.length)}%）`);
  return { X, Y };
}

/* ---------- 3. CART 回歸樹擬合（O(n log n) 增量 SSE 掃描） ---------- */
function mean(arr, idxs) { let s = 0; for (const i of idxs) s += arr[i]; return s / idxs.length; }
function sse(arr, idxs, m) { let s = 0; for (const i of idxs) { const d = arr[i] - m; s += d * d; } return s; }

function fitTree(X, Y, opts) {
  function build(idxs, depth) {
    const n = idxs.length;
    if (n < 2) return { leaf: true, value: n ? Y[idxs[0]] : 0 };
    const m = mean(Y, idxs);
    const err = sse(Y, idxs, m);
    if (depth >= opts.maxDepth || n < opts.minSamplesSplit || err < 1e-9) return { leaf: true, value: m };
    let best = null;
    for (const f of opts.features) {
      const sorted = idxs.slice().sort((a, b) => X[a][f] - X[b][f]);
      let sumL = 0, sumsqL = 0, nL = 0;
      let sumR = 0, sumsqR = 0, nR = sorted.length;
      for (const i of sorted) { sumR += Y[i]; sumsqR += Y[i] * Y[i]; }
      for (let k = 0; k < sorted.length - 1; k++) {
        const i = sorted[k];
        sumL += Y[i]; sumsqL += Y[i] * Y[i]; nL++; nR--;
        sumR -= Y[i]; sumsqR -= Y[i] * Y[i];
        const xk = X[i][f], xk1 = X[sorted[k + 1]][f];
        if (xk === xk1) continue; // 同值不切
        const mL = sumL / nL, mR = sumR / nR;
        const gain = err - ((sumsqL - nL * mL * mL) + (sumsqR - nR * mR * mR));
        if (!best || gain > best.gain) {
          best = { gain, f, th: (xk + xk1) / 2, leftIdx: sorted.slice(0, k + 1), rightIdx: sorted.slice(k + 1) };
        }
      }
    }
    if (!best || best.gain <= 1e-9) return { leaf: true, value: m };
    return { feature: best.f, threshold: best.th, left: build(best.leftIdx, depth + 1), right: build(best.rightIdx, depth + 1) };
  }
  return build([...Array(X.length).keys()], 0);
}

// 把巢狀樹攤平成 sklearn 風格的平行陣列
function toSklearn(node) {
  const children_left = [], children_right = [], feature = [], threshold = [], value = [];
  (function rec(n) {
    const idx = children_left.length;
    if (n.leaf) { children_left.push(-1); children_right.push(-1); feature.push(-2); threshold.push(0); value.push(n.value); return; }
    children_left.push(-1); children_right.push(-1); feature.push(n.feature); threshold.push(n.threshold); value.push(0);
    const li = children_left.length; rec(n.left); const ri = children_left.length; rec(n.right);
    children_left[idx] = li; children_right[idx] = ri;
  })(node);
  return { children_left, children_right, feature, threshold, value };
}

function bootstrap(n) {
  const idxs = [];
  for (let i = 0; i < n; i++) idxs.push(Math.floor(Math.random() * n));
  return idxs;
}

/* ---------- 4. 評分（與平台一致：greedy，predict>0 → 拍翅膀） ---------- */
function evaluateModel(model, scoreFn) {
  const sim = new FlappySim(CONFIG.EVAL_SEED);
  let y = CONFIG.HEIGHT / 2, vy = 0, frame = 0;
  while (frame < CONFIG.MAX_FRAMES) {
    const inputs = sim.getInputs(y);
    if (scoreFn(inputs) > 0) vy = CONFIG.FLAP_VY; else vy += CONFIG.GRAVITY;
    y += vy;
    if (sim.collisionAt(y)) break;
    sim.advance();
    frame++;
  }
  return sim.score;
}
function treeScore(tree, inputs) {
  let i = 0;
  while (tree.children_left[i] !== -1 && tree.children_right[i] !== -1) {
    i = (inputs[tree.feature[i]] <= tree.threshold[i]) ? tree.children_left[i] : tree.children_right[i];
  }
  return tree.value[i];
}

function main() {
  const args = parseArgs(process.argv);
  if (args.mode !== 'tree' && args.mode !== 'forest') {
    console.error('錯誤：--mode 必須是 tree 或 forest'); process.exit(1);
  }
  const t0 = Date.now();

  console.log('步驟 1/4：訓練神經演化老師…');
  const teacher = trainTeacher(args);
  console.log('步驟 2/4：蒐集訓練資料…');
  const { X, Y } = collectDataset(teacher, args);

  let model;
  if (args.mode === 'tree') {
    console.log(`步驟 3/4：CART 擬合決策樹（maxDepth=${args.maxDepth}）…`);
    const tree = toSklearn(fitTree(X, Y, { maxDepth: args.maxDepth, minSamplesSplit: args.minSamplesSplit, features: [0, 1] }));
    model = { type: 'decision_tree', tree };
  } else {
    console.log(`步驟 3/4：CART 擬合隨機森林（${args.nTrees} 棵, maxDepth=${args.maxDepth}）…`);
    const trees = [];
    for (let t = 0; t < args.nTrees; t++) {
      const idxs = bootstrap(X.length);
      const Xb = idxs.map(i => X[i]), Yb = idxs.map(i => Y[i]);
      // 本問題只有 2 個特徵，隨機森林的特徵子抽樣若只取 1 個會過弱；
      // 故每棵樹都用全部特徵，多樣性來自 bootstrap 子樣本（bagging）。
      trees.push(toSklearn(fitTree(Xb, Yb, { maxDepth: args.maxDepth, minSamplesSplit: Math.max(4, args.minSamplesSplit), features: [0, 1] })));
    }
    model = { type: 'random_forest', trees };
  }

  console.log('步驟 4/4：用評分賽道驗證…');
  const score = (args.mode === 'tree')
    ? evaluateModel(model, x => treeScore(model.tree, x))
    : evaluateModel(model, x => { let s = 0; for (const tr of model.trees) s += treeScore(tr, x); return s / model.trees.length; });

  fs.writeFileSync(args.out, JSON.stringify(model, null, 2));
  const secs = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`\n完成（耗時 ${secs}s）。${args.mode === 'tree' ? '決策樹' : '隨機森林'}評分賽道分數=${score}，模型已存至 ${args.out}`);
  console.log(`提交：curl -X POST http://localhost:3000/api/submit -H "Content-Type: application/json" -d '{"name":"sklearn-agent","model":'$(cat ${args.out})'}'`);
}

main();
