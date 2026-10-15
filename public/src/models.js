/**
 * 統一模型抽象（UMD：Node require 與瀏覽器 <script> 共用，零外部依賴）
 *
 * 本平台支援多種「玩 Flappy Bird 的模型」，但對外只有一個統一契約：
 *   const m = Model.fromJSON(spec);
 *   const flap = m.predict([dx, dy]) > m.threshold;   // true → 拍翅膀，false → 不動
 *
 * 支援的模型類型（type 欄位）：
 *   - neuralnet      2→6→1 前饋神經網絡（既有，threshold 預設 0.5）
 *   - decision_tree  sklearn 風格回歸決策樹（threshold 預設 0）
 *   - random_forest  多棵決策樹（bagging）平均（threshold 預設 0）
 *   - q_table        強化學習 tabular Q-learning 查表策略（threshold 預設 0 = 選 Q 值較高的動作）
 *
 * 向後相容：沒有 type 但帶有 arch 的舊模型 → 視為 neuralnet。
 *
 * 重點：predict / threshold 在 Node 評分與瀏覽器觀戰必須完全一致，
 * 因此本檔案是「評分」與「觀戰」唯一真理來源，兩端都用同一份程式碼。
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory(require('./neuralnet.js'));
  } else {
    root.Model = factory(root.NeuralNet).Model;
  }
})(typeof self !== 'undefined' ? self : this, function (NeuralNetModule) {
  'use strict';

  const NeuralNet = NeuralNetModule && NeuralNetModule.NeuralNet ? NeuralNetModule.NeuralNet : NeuralNetModule;

  function clampInt(v, lo, hi, dflt) {
    let n = Math.floor(Number(v));
    if (!Number.isFinite(n)) n = dflt;
    return Math.max(lo, Math.min(hi, n));
  }

  /* ============ 1. neuralnet ============ */
  function neuralnetModel(spec) {
    const net = NeuralNet.deserialize(spec);
    const threshold = (spec.threshold != null) ? Number(spec.threshold) : 0.5;
    return {
      type: 'neuralnet',
      threshold,
      predict: x => net.activate(x)[0]
    };
  }

  /* ============ 2. 決策樹（sklearn 風格：children_left/right/feature/threshold/value） ============ */
  // 單棵樹推論：從根（index 0）往下走，遇到 children_left===-1 即為葉節點。
  function treePredict(tree, x) {
    let i = 0;
    const { children_left: L, children_right: R, feature: F, threshold: T, value: V } = tree;
    while (L[i] !== -1 && R[i] !== -1) {
      i = (x[F[i]] <= T[i]) ? L[i] : R[i];
    }
    const v = V[i];
    return Array.isArray(v) ? v[0] : v; // 葉輸出（回歸值，純量）
  }

  function decisionTreeModel(spec) {
    const tree = spec.tree;
    if (!tree || !Array.isArray(tree.children_left)) {
      throw new Error('decision_tree 缺少 tree（children_left/right/feature/threshold/value）');
    }
    const threshold = (spec.threshold != null) ? Number(spec.threshold) : 0;
    return { type: 'decision_tree', threshold, predict: x => treePredict(tree, x) };
  }

  function randomForestModel(spec) {
    if (!Array.isArray(spec.trees) || !spec.trees.length) {
      throw new Error('random_forest 缺少 trees[]');
    }
    const trees = spec.trees;
    const threshold = (spec.threshold != null) ? Number(spec.threshold) : 0;
    return {
      type: 'random_forest',
      threshold,
      predict: x => {
        let s = 0;
        for (let k = 0; k < trees.length; k++) s += treePredict(trees[k], x);
        return s / trees.length;
      }
    };
  }

  /* ============ 3. q_table（強化學習） ============ */
  // 狀態離散化：把連續輸入 [dx, dy]（約落在 [-1,1]）切成 bins[x] × bins[y] 格。
  function discretize(x, bins) {
    const bx = clampInt(((x[0] + 1) / 2) * bins[0], 0, bins[0] - 1, 0);
    const by = clampInt(((x[1] + 1) / 2) * bins[1], 0, bins[1] - 1, 0);
    return [bx, by];
  }

  function qTableModel(spec) {
    const bins = spec.bins && spec.bins.length === 2 ? spec.bins : [10, 10];
    const q = spec.q;
    if (!Array.isArray(q)) throw new Error('q_table 缺少 q 陣列');
    const threshold = (spec.threshold != null) ? Number(spec.threshold) : 0;
    return {
      type: 'q_table',
      threshold,
      predict: x => {
        const [bx, by] = discretize(x, bins);
        const cell = q[bx] && q[bx][by] ? q[bx][by] : [0, 0];
        // 回傳「拍翅膀」相對「不動」的優勢（advantage）；>0 表示拍翅膀較好。
        return cell[1] - cell[0];
      }
    };
  }

  /* ============ 工廠 ============ */
  function fromJSON(spec) {
    if (!spec || typeof spec !== 'object') throw new Error('模型必須是 JSON 物件');
    const type = spec.type || (spec.arch ? 'neuralnet' : null);
    if (!type) throw new Error('無法辨識模型類型（需有 type 欄位，或舊格式 arch）');
    switch (type) {
      case 'neuralnet': return neuralnetModel(spec);
      case 'decision_tree': return decisionTreeModel(spec);
      case 'random_forest': return randomForestModel(spec);
      case 'q_table': return qTableModel(spec);
      default: throw new Error('不支援的模型類型：' + type + '（支援：neuralnet / decision_tree / random_forest / q_table）');
    }
  }

  // 模型類型 → 中文顯示名（前端排行榜徽章用）
  const TYPE_LABELS = {
    neuralnet: '🧠 神經網絡',
    decision_tree: '🌳 決策樹',
    random_forest: '🌲 隨機森林',
    q_table: '🎯 強化學習'
  };
  function label(type) { return TYPE_LABELS[type] || type; }

  return { Model: { fromJSON, label, TYPE_LABELS } };
});
