/**
 * 瀏覽器端神經演化訓練器（與 src/train.js 同一套 GA 邏輯）
 * - 供「在線訓練」頁視覺化使用：外部迴圈每幀驅動鳥群，鳥全滅時呼叫 nextGeneration(fits) 結算並繁殖
 * - 依賴全域 NeuralNet / FlappySim / FlappyConfig（由 /src/*.js 注入）
 */
(function (root) {
  'use strict';
  const NeuralNet = root.NeuralNet;
  const CONFIG = root.FlappyConfig;

  // 單隻鳥前進一幀的 fitness 增量：存活 + 接近缺口中心（與 train.js 一致，提供演化梯度）
  function fitnessDelta(inputs) {
    return 1 + Math.max(0, 1 - Math.abs(inputs[1]));
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

  class Trainer {
    /**
     * @param {{generations:number, population:number, topK:number, mutationRate:number, seed:number}} opts
     */
    constructor(opts) {
      this.arch = [2, 6, 1]; // 平台提交規定架構
      this.generations = Math.max(1, opts.generations | 0);
      this.population = Math.max(2, opts.population | 0);
      this.topK = Math.max(1, Math.min(opts.topK | 0, this.population));
      this.mutationRate = opts.mutationRate;
      // seed 僅供記錄/復現提示；演化本身用 Math.random（與 train.js 一致）
      this.seed = opts.seed;
      this.pop = [];
      for (let i = 0; i < this.population; i++) this.pop.push(NeuralNet.random(this.arch));
      this.gen = 0;
      this.best = null;
      this.bestFitness = -1;
      this.history = [];
    }

    /**
     * 用本代每隻鳥的 fitness 結算：記錄、追蹤最佳、繁殖下一代。
     * @param {number[]} fits 與 this.pop 等長的 fitness 陣列
     * @returns {{gen:number,total:number,best:number,avg:number,topScores:number[],bestFitness:number,done:boolean,earlyStop:boolean}}
     */
    nextGeneration(fits) {
      const ranked = this.pop
        .map((net, i) => ({ net, score: fits[i] }))
        .sort((x, y) => y.score - x.score);

      if (ranked[0].score > this.bestFitness) {
        this.bestFitness = ranked[0].score;
        this.best = ranked[0].net;
      }

      const avg = fits.reduce((s, v) => s + v, 0) / fits.length;
      const top = ranked.slice(0, this.topK);
      this.history.push({ gen: this.gen + 1, best: ranked[0].score, avg });

      const earlyStop = this.bestFitness >= 3000;
      const done = earlyStop || (this.gen + 1) >= this.generations;

      if (!done) {
        const next = top.map(t => t.net); // 菁英直接保留
        while (next.length < this.population) {
          const p1 = top[Math.floor(Math.random() * top.length)].net;
          const p2 = top[Math.floor(Math.random() * top.length)].net;
          next.push(mutate(crossover(p1, p2), this.mutationRate));
        }
        this.pop = next;
      }
      this.gen += 1;

      return {
        gen: this.gen,
        total: this.generations,
        best: ranked[0].score,
        avg,
        topScores: top.map(t => t.score),
        bestFitness: this.bestFitness,
        done,
        earlyStop
      };
    }

    /** 最佳模型的深拷貝 JSON（避免後續突變汙染） */
    model() {
      if (!this.best) return null;
      return JSON.parse(JSON.stringify(this.best.serialize()));
    }
  }

  root.FlappyTrainer = { Trainer, crossover, mutate, fitnessDelta, EVAL_SEED: CONFIG.EVAL_SEED };
})(typeof self !== 'undefined' ? self : this);
