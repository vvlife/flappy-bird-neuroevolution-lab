/**
 * 輕量前饋神經網絡（無外部依賴）
 * - 可在 Node (require) 與瀏覽器 (<script>) 共用（UMD 包裝）
 * - 預設架構 2 -> 6 -> 1：輸入[水平距離, 垂直差距]，輸出>0.5 拍翅膀
 * - 隱藏層用 tanh，輸出層用 sigmoid（輸出落在 0~1，便於 0.5 閾值決策）
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = { NeuralNet: factory() };
  } else {
    root.NeuralNet = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

  class NeuralNet {
    constructor(arch) {
      this.arch = arch.slice();
      this.weights = [];
      this.biases = [];
    }

    static random(arch) {
      const net = new NeuralNet(arch);
      net.randomize();
      return net;
    }

    randomize() {
      this.weights = [];
      this.biases = [];
      for (let l = 0; l < this.arch.length - 1; l++) {
        const rows = this.arch[l + 1];
        const cols = this.arch[l];
        const w = [];
        for (let i = 0; i < rows; i++) {
          const row = [];
          for (let j = 0; j < cols; j++) row.push(Math.random() * 2 - 1);
          w.push(row);
        }
        this.weights.push(w);
        const b = [];
        for (let i = 0; i < rows; i++) b.push(Math.random() * 2 - 1);
        this.biases.push(b);
      }
    }

    activate(input) {
      let a = input.slice();
      for (let l = 0; l < this.weights.length; l++) {
        const w = this.weights[l];
        const b = this.biases[l];
        const isLast = (l === this.weights.length - 1);
        const next = new Array(w.length);
        for (let j = 0; j < w.length; j++) {
          let sum = b[j];
          const wj = w[j];
          for (let i = 0; i < wj.length; i++) sum += wj[i] * a[i];
          next[j] = isLast ? sigmoid(sum) : Math.tanh(sum);
        }
        a = next;
      }
      return a;
    }

    serialize() {
      return { arch: this.arch.slice(), weights: this.weights, biases: this.biases };
    }

    static deserialize(obj) {
      const net = new NeuralNet(obj.arch);
      net.weights = obj.weights;
      net.biases = obj.biases;
      return net;
    }
  }

  return NeuralNet;
});
