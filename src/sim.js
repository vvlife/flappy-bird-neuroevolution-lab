/**
 * Flappy Bird 遊戲世界（無 DOM 依賴，可 headless 評分，也可在前端視覺化）
 * - 管柱（pipes）為全域狀態，所有鳥共用同一條賽道 → 評分公平
 * - getInputs(birdY) / collisionAt(birdY) / advance() 由外部驅動
 *
 * 評分賽道使用固定種子 EVAL_SEED，確保「提交時的分數」與「前端即時播放的分數」一致。
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    const ns = factory();
    root.FlappySim = ns.FlappySim;
    root.FlappyConfig = ns.CONFIG;
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const CONFIG = {
    WIDTH: 400,
    HEIGHT: 600,
    BIRD_X: 80,
    BIRD_SIZE: 20,
    GRAVITY: 0.45,
    FLAP_VY: -8,
    PIPE_SPEED: 3,
    PIPE_WIDTH: 60,
    PIPE_GAP: 140,
    GAP_MIN: 90,        // 缺口中心距頂部最小距離
    GAP_MAX: 510,       // 缺口中心距頂部最大距離 (HEIGHT - 90)
    SPAWN_INTERVAL: 90, // 每隔多少 frame 生成一根管柱
    EVAL_SEED: 12345,   // 評分賽道固定種子
    MAX_FRAMES: 20000   // 單次評分最長 frame 數（防無限）
  };

  // 可複現的偽隨機數（mulberry32）
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  class FlappySim {
    constructor(seed) {
      this.reset(seed);
    }

    reset(seed) {
      this.seed = (seed === undefined) ? CONFIG.EVAL_SEED : seed;
      this.rng = mulberry32(this.seed);
      this.frame = 0;
      this.score = 0;
      this.pipes = [];
      this._spawnTimer = 0;
      this._spawnPipe();
    }

    _spawnPipe() {
      const gapY = CONFIG.GAP_MIN + this.rng() * (CONFIG.GAP_MAX - CONFIG.GAP_MIN);
      this.pipes.push({ x: CONFIG.WIDTH, gapY: gapY, passed: false });
    }

    // 鳥固定在 BIRD_X，回傳 [水平距離(歸一化), 與缺口中心的垂直差距(歸一化)]
    getInputs(birdY) {
      let np = null;
      for (const p of this.pipes) {
        if (p.x + CONFIG.PIPE_WIDTH > CONFIG.BIRD_X - CONFIG.BIRD_SIZE / 2) { np = p; break; }
      }
      if (!np) return [1.0, 0.0];
      const dx = (np.x - CONFIG.BIRD_X) / CONFIG.WIDTH;
      const dy = (birdY - np.gapY) / CONFIG.HEIGHT;
      return [dx, dy];
    }

    collisionAt(birdY) {
      const half = CONFIG.BIRD_SIZE / 2;
      if (birdY - half < 0) return true;
      if (birdY + half > CONFIG.HEIGHT) return true;
      for (const p of this.pipes) {
        if (CONFIG.BIRD_X + half > p.x && CONFIG.BIRD_X - half < p.x + CONFIG.PIPE_WIDTH) {
          const gapTop = p.gapY - CONFIG.PIPE_GAP / 2;
          const gapBottom = p.gapY + CONFIG.PIPE_GAP / 2;
          if (birdY - half < gapTop || birdY + half > gapBottom) return true;
        }
      }
      return false;
    }

    advance() {
      for (const p of this.pipes) {
        p.x -= CONFIG.PIPE_SPEED;
        if (!p.passed && p.x + CONFIG.PIPE_WIDTH < CONFIG.BIRD_X) {
          p.passed = true;
          this.score += 1;
        }
      }
      this.pipes = this.pipes.filter(p => p.x + CONFIG.PIPE_WIDTH > -10);
      this._spawnTimer += 1;
      if (this._spawnTimer >= CONFIG.SPAWN_INTERVAL) {
        this._spawnTimer = 0;
        this._spawnPipe();
      }
      this.frame += 1;
    }
  }

  return { FlappySim, CONFIG, mulberry32 };
});
