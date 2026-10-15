# Flappy Bird AI 競賽平台 · 接口文檔

本平台讓學生（或 WorkBuddy 之類的 AI 工具）訓練一個「玩 Flappy Bird 的神經網絡」，
把訓練好的模型提交上來，伺服器會用**同一條固定賽道**幫它打分，並登上線上排行榜；
任何人都能在網頁上**即時觀看**任意一筆提交實際飛行的畫面。

設計原則：**訓練在哪裡發生由你決定**（本地脚本、Notebook、另一台機器、AI 代理都行），
本平台只負責「收模型 → 跑結果 → 記分 → 展示」。

---

## 1. 模型格式（Model Schema）

提交的是一個 JSON 物件，描述一個 **2 → 6 → 1** 的前饋神經網絡：

```json
{
  "arch": [2, 6, 1],
  "weights": [
    [ [w,w,w,w,w,w], [w,w,w,w,w,w] ],
    [ [w], [w], [w], [w], [w], [w] ]
  ],
  "biases": [
    [b, b, b, b, b, b],
    [b]
  ]
}
```

- `arch`：固定為 `[2, 6, 1]`（輸入層 2、隱藏層 6、輸出層 1）。
- `weights`：長度 = 層數 - 1 = 2。
  - `weights[0]` 是「輸入→隱藏」矩陣，形狀 `6 × 2`（6 個隱藏神經元 × 2 個輸入）。
  - `weights[1]` 是「隱藏→輸出」矩陣，形狀 `1 × 6`。
- `biases`：長度 = 2。`biases[0]` 形狀 `6`，`biases[1]` 形狀 `1`。

### 前向計算（推論）

對於輸入 `x = [dx, dy]`：

```
h_j = tanh( sum_i(weights[0][j][i] * x[i]) + biases[0][j] )      // 隱藏層，j = 0..5
o   = sigmoid( sum_j(weights[1][0][j] * h_j) + biases[1][0] )    // 輸出層，o ∈ (0,1)
action = o > 0.5 ? 拍翅膀 : 不動
```

### 輸入含義（與本平台仿真一致）

- `dx` =（最近管柱左緣 x − 鳥的 x）/ 畫布寬度 → 歸一化水平距離
- `dy` =（鳥的 y − 缺口中心 y）/ 畫布高度 → 歸一化垂直差距

> 只要你的模型輸出的 `dx/dy` 定義與上述一致，分數就具可比性。
> 參考實作見 `src/neuralnet.js` 與 `src/sim.js`（Node / 瀏覽器共用，純 JS 無依賴）。

---

## 2. 端點（Endpoints）

基礎網址：`http://localhost:3000`（部署後請換成你的網域）

### `GET /api/leaderboard`

回傳完整排行榜（由高到低排序，含每筆的 `model`，便於前端重播）。

```bash
curl http://localhost:3000/api/leaderboard
```

回傳範例：

```json
[
  {
    "id": "m1a2b3c",
    "name": "學生-A",
    "score": 87,
    "arch": [2, 6, 1],
    "model": { "arch": [2,6,1], "weights": [...], "biases": [...] },
    "createdAt": "2026-10-14T10:30:00.000Z"
  }
]
```

### `POST /api/submit`

提交一個訓練好的模型。伺服器用固定種子賽道（預設 `EVAL_SEED=12345`）跑出分數，寫入排行榜。

- 要求標頭：`Content-Type: application/json`
- 請求本體：

```json
{ "name": "學生-A", "model": { "arch": [2,6,1], "weights": [...], "biases": [...] } }
```

- 成功回傳（HTTP 200）：

```json
{ "id": "m1a2b3c", "score": 87, "rank": 1, "total": 5 }
```

- 錯誤：
  - `400`：`name`/`model` 缺漏、`arch` 非 `[2,6,1]`、或模型計算失敗。

```bash
# 用檔案提交
curl -X POST http://localhost:3000/api/submit \
  -H "Content-Type: application/json" \
  -d '{"name":"學生-A","model":'$(cat model.json)'}'
```

### `GET /api/model/:id`

回傳單筆提交的模型 JSON（前端「即時畫面」即時載入用）。

```bash
curl http://localhost:3000/api/model/m1a2b3c
```

### `GET /api/benchmark/:id`

對同一模型在**多個種子**賽道上跑分，衡量穩健度（官方賽道以外的表現）。
第一種子固定為官方 `EVAL_SEED`，其餘為固定替代種子 → 結果可複現。

- 參數：`count`（種子數量，1–50，預設 10）

```bash
curl "http://localhost:3000/api/benchmark/m1a2b3c?count=10"
```

回傳範例：

```json
{
  "id": "m1a2b3c",
  "name": "學生-A",
  "evalSeed": 12345,
  "seedsCount": 10,
  "results": [
    { "seed": 12345, "score": 87, "frames": 3200, "official": true, "perfect": false }
  ],
  "summary": { "avg": 42.3, "best": 87, "worst": 12, "median": 40, "perfectRuns": 1, "survivalRate": 10 },
  "generatedAt": "2026-10-14T10:30:00.000Z"
}
```

- `results[].perfect`：撐滿 `MAX_FRAMES`（極強模型）。
- 網頁上每列有「📊」可即時看此統計，並用「⬇ 下載 JSON」把整份報告存檔。

---

## 3. 評分規則（Score）

- 評分賽道使用**固定隨機種子**，所有人的模型面對完全相同的管柱序列 → 公平可比。
- 分數 = 成功通過的管柱數（pipe 越過鳥之後 +1）。
- 單次評分最長 `MAX_FRAMES = 20000` frame，避免無限飛行；若撐滿則以該上限計（極強模型）。
- 前端「即時畫面」使用**同一個種子**播放，所以你看到的畫面分數會與提交時一致。

---

## 4. 如何訓練（三種路徑）

### 路徑 A：用本平台提供的訓練腳本（神經演化 / 遺傳演算法）

```bash
node src/train.js --generations 40 --population 20 --topK 4 --mutationRate 0.1 --out model.json
# 訓練完會印出最佳分數，並產出 model.json
```

原理：每一代讓一群隨機神經網絡在同一条賽道同時飛，依通過管柱數選出強者，
透過「菁英保留 + 交叉 + 突變」產生下一代，反覆演化。

### 路徑 B：自己用任何語言寫訓練器

只要最終產出符合第 1 節的 JSON 即可。你可以：
- 用 Python（PyTorch / NumPy）自己實作演化或強化學習；
- 直接用 `src/sim.js` 裡的 `FlappySim` 邏輯在你的語言重寫一個仿真當訓練環境；
- 任何能輸出 `arch/weights/biases` 的模型都接受。

### 路徑 C：讓 AI 工具（如 WorkBuddy）代勞

把本文件（與 `README.md`）交給 WorkBuddy 之類的代理，請它：
1. 閱讀 `src/train.js` / `src/neuralnet.js` / `src/sim.js` 理解環境；
2. 執行訓練腳本產出 `model.json`（或自行撰寫訓練程式）；
3. 呼叫 `POST /api/submit` 把模型送上台；
4. 回報得到的 `score` 與 `rank`。

範例（WorkBuddy 可在對話中執行的 shell 指令）：

```bash
# 1) 訓練
node src/train.js --generations 40 --population 20 --out model.json
# 2) 提交
node src/submit.js model.json "WorkBuddy-Agent"
```

---

## 5. 網頁介面

首頁頂端有選單（菜單）可切換三個分頁：

### 🏟 排行榜 / 競技

- 左側排行榜，每筆有「▶」播放該模型實飛、「📊」跑 **benchmark** 並下載 JSON 報告；
- 勾選多列 →「🏁 一起玩」：多個 Agent 在**同一條評分賽道**同時競技，各自彩色鳥與即時分數看板，結束顯示冠軍；
- 畫面支援 **1x / 2x / 4x / 8x 倍速**；
- 「觀看最佳 / 隨機觀看 / 刷新」按鈕；
- 「🤖 給 Agent 的說明」可**一鍵複製**，交給 WorkBuddy 之類的工具即可照著訓練與提交；
- 「📤 手動提交模型」可貼 JSON 或上傳 `model.json` 直接試送。

### 🧬 在線訓練（瀏覽器內訓練，無需安裝任何東西）

在瀏覽器裡直接跑神經演化，可即時調節參數並觀察種群競爭的動畫：

| 參數 | 說明 | 預設 |
|---|---|---|
| 代數 | 演化總代數（達成目標會提前結束） | 30 |
| 種群數量 | 每代鳥的數量 | 20 |
| 菁英保留 topK | 每代直接保留的強者數 | 4 |
| 突變率 | 每個權重被隨機擾動的機率 | 0.1 |
| 隨機種子 | 記錄用（演化本身用亂數） | 12345 |
| 每代最長幀數 | 單代 simulate 的 frame 上限 | 2000 |
| 動畫速度 | 每個動畫幀跑幾個 sim step | 15 |

- 完成後顯示**評分賽道實際分數**、演化代數與最佳 fitness，並附「提交到排行榜 / 下載 model.json / 觀看 / 🔁 再跑一次」。
- 訓練在背景執行，切換到其他分頁不會中斷。
- 產生的模型格式與 CLI 完全一致，可直接 `POST /api/submit`。

> 演化具隨機性，單次結果有方差；分數不理想時按「🔁 再跑一次」即可。

### 📖 API 文檔

本文件（`docs/API.md`）在頁面上即時渲染，並附「📋 複製 API.md 原文」按鈕。
