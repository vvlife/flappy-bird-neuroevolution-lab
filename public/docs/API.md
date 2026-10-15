# Flappy Bird AI 競賽平台 · 接口文檔

本平台讓學生（或 WorkBuddy 之類的 AI 工具）訓練一個「玩 Flappy Bird 的神經網絡」，
把訓練好的模型提交上來，伺服器會用**同一條固定賽道**幫它打分，並登上線上排行榜；
任何人都能在網頁上**即時觀看**任意一筆提交實際飛行的畫面。

設計原則：**訓練在哪裡發生由你決定**（本地脚本、Notebook、另一台機器、AI 代理都行），
本平台只負責「收模型 → 跑結果 → 記分 → 展示」。

---

## 1. 模型格式（Model Schema）

平台支援**多種模型類型**，但對外只有一個統一契約（參見 `public/src/models.js`，Node 評分與瀏覽器觀戰共用同一份程式碼）：

```js
const m = Model.fromJSON(spec);          // spec = 你提交的 model 物件
const flap = m.predict([dx, dy]) > m.threshold;   // true → 拍翅膀，false → 不動
```

- 所有模型都吃同一組**連續輸入** `x = [dx, dy]`（見下方「輸入含義」）。
- 統一用 `predict(x) > threshold` 決策；`threshold` 每種類型有預設值，但都可被 `spec.threshold` 覆寫。
- 用 `type` 欄位區分模型類型。**舊格式（只有 `arch` 沒有 `type`）會自動視為 `neuralnet`**，完全向後相容。

支援四種 `type`：

| `type` | 中文徽章 | 預設 `threshold` | 說明 |
|--------|----------|------------------|------|
| `neuralnet` | 🧠 神經網絡 | `0.5` | 2→6→1 前饋神經網絡（既有，演化/梯度都能產出） |
| `decision_tree` | 🌳 決策樹 | `0` | sklearn 風格回歸樹（children_left/right/feature/threshold/value） |
| `random_forest` | 🌲 隨機森林 | `0` | 多棵決策樹（bagging）平均輸出 |
| `q_table` | 🎯 強化學習 | `0` | tabular Q-learning 查表策略（離散化狀態） |

### 1.1 神經網絡 `neuralnet`（2 → 6 → 1）

```json
{
  "type": "neuralnet",
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
- 前向計算：

```
h_j = tanh( sum_i(weights[0][j][i] * x[i]) + biases[0][j] )      // 隱藏層，j = 0..5
o   = sigmoid( sum_j(weights[1][0][j] * h_j) + biases[1][0] )    // 輸出層，o ∈ (0,1)
action = o > 0.5 ? 拍翅膀 : 不動
```

### 1.2 決策樹 `decision_tree`（sklearn 風格）

```json
{
  "type": "decision_tree",
  "tree": {
    "children_left":  [ 1, -1, -1, ... ],
    "children_right": [ 2, -1, -1, ... ],
    "feature":        [ 0, -2, -2, ... ],
    "threshold":      [ 0.12, 0, 0, ... ],
    "value":          [ 0, 0.8, -0.3, ... ]
  }
}
```

- 平行陣列（與 scikit-learn `DecisionTreeRegressor.tree_` 同構）：
  - `children_left[i]` / `children_right[i]`：節點 `i` 的左右子節點索引；等於 `-1` 表示葉節點。
  - `feature[i]`：內部節點用來切分的特徵索引（`0`=dx, `1`=dy）；葉節點用 `-2`。
  - `threshold[i]`：切分閾值；`x[feature[i]] <= threshold[i]` 走左，否則走右。
  - `value[i]`：葉節點的回歸輸出（純量）；內部節點為 `0`。
- 推論：從根（index 0）往下走，直到 `children_left===-1` 的葉節點，回傳其 `value`。
- 決策：`value > 0` → 拍翅膀。

> 想從 Python 直接產出？參考 `src/train_sklearn.js` 的 `toSklearn()`，它把任意遞迴樹攤平成這組平行陣列；
> scikit-learn 也提供 `tree_` 的平行陣列，照同結構填即可。

### 1.3 隨機森林 `random_forest`

```json
{
  "type": "random_forest",
  "trees": [ { "children_left":[...], "children_right":[...], "feature":[...], "threshold":[...], "value":[...] }, ... ]
}
```

- `trees`：若干棵 `decision_tree`（格式同上 §1.2）。
- 推論：對每棵樹取葉輸出，取**平均**作為 `predict`。
- 決策：平均 `> 0` → 拍翅膀。

### 1.4 強化學習 `q_table`（tabular Q-learning）

```json
{
  "type": "q_table",
  "bins": [30, 30],
  "q": [
    [ [0.0, 0.3], [0.1, 0.2], ... ],   // 第 0 個 x 格
    [ [0.0, 0.3], ... ],               // 第 1 個 x 格
    ...
  ]
}
```

- `bins`：`[binsX, binsY]`，把連續輸入 `x=[dx,dy]`（約落在 `[-1,1]`）離散化成的格子數。
- `q[bx][by]`：`[Q(不動), Q(拍翅膀)]` 兩個動作價值。`bx = clamp(((dx+1)/2)*binsX, 0, binsX-1)`，`by` 同理。
- 推論：回傳 `q[bx][by][1] - q[bx][by][0]`（拍翅膀相對不動的**優勢**）。
- 決策：優勢 `> 0` → 拍翅膀（即選 Q 值較高的動作）。

> 關鍵細節：二值「拍/不拍」控制需要**細離散化**（建議 `bins≥20`，本平台訓練預設 `30×30`）才能把鳥穩定維持在 ±60px 缺口內；
> 太粗（如 `12×12`）會完全飛不過第一根管柱（分數 0）。

### 輸入含義（四種模型皆相同，與本平台仿真一致）

- `dx` =（最近管柱左緣 x − 鳥的 x）/ 畫布寬度 → 歸一化水平距離
- `dy` =（鳥的 y − 缺口中心 y）/ 畫布高度 → 歸一化垂直差距

> 只要你的模型輸出的 `dx/dy` 定義與上述一致，分數就具可比性。
> 參考實作見 `public/src/neuralnet.js`、`public/src/sim.js` 與 `public/src/models.js`（Node / 瀏覽器共用，純 JS 無依賴）。

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
    "type": "neuralnet",
    "arch": [2, 6, 1],
    "model": { "type": "neuralnet", "arch": [2,6,1], "weights": [...], "biases": [...] },
    "createdAt": "2026-10-14T10:30:00.000Z"
  }
]
```

### `POST /api/submit`

提交一個訓練好的模型。伺服器用固定種子賽道（預設 `EVAL_SEED=12345`）跑出分數，寫入排行榜。

- 要求標頭：`Content-Type: application/json`
- 請求本體：

```json
{ "name": "學生-A", "model": { "type": "neuralnet", "arch": [2,6,1], "weights": [...], "biases": [...] } }
```

  > `model.type` 支援 `neuralnet` / `decision_tree` / `random_forest` / `q_table`（見 §1）。
  > 省略 `type` 但帶 `arch` 的舊格式會自動視為 `neuralnet`。

- 成功回傳（HTTP 200）：

```json
{ "id": "m1a2b3c", "score": 87, "rank": 1, "total": 5 }
```

- 錯誤：
  - `400`：`name`/`model` 缺漏、`type` 不在支援清單內、或模型計算失敗（伺服器會先做一次 `Model.fromJSON` + `predict([0.1,0.2])` 冒煙測試）。

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

### 路徑 D：強化學習（tabular Q-learning，產出 `q_table`）

```bash
node src/train_rl.js --episodes 3000 --bins 30 --epsStart 0.2 --epsEnd 0.0 \
  --epsDecay 300 --trainFrames 2000 --alpha 0.1 --gamma 0.99 --out rl-model.json
```

- 原理：在**固定評分賽道**（`EVAL_SEED`）上，ε-greedy 探索 + TD(0) 更新 `Q(s,a)`。
- 獎勵：通過管柱 `+1`、存活一步 `+0.1`、靠近缺口中心有小塑形獎勵；死亡 `-1`（刻意保持很小，避免 TD 把全部 Q 拖成深度負數）。
- 探索率 `ε` 在前 `epsDecay` 回合從 `epsStart` 線性衰減到 `epsEnd`，之後**純貪婪**——這一步是能飛過管柱的關鍵（持續探索會讓死亡經驗污染 Q 表）。
- 輸出 `{ type:"q_table", bins, q }`，預設參數在評分賽道可得滿分（221）。
- 重要：**必須在 `EVAL_SEED` 上訓練**，隨機種子賽道只會讓鳥一直死、收不到正向回報。

### 路徑 E：模仿學習產出決策樹 / 隨機森林（`decision_tree` / `random_forest`）

```bash
# 單棵 CART 決策樹
node src/train_sklearn.js --mode tree --episodes 25 --maxSteps 1000 --out tree-model.json

# 隨機森林（10 棵，bootstrap 抽樣）
node src/train_sklearn.js --mode forest --episodes 25 --maxSteps 1000 --out forest-model.json
```

- 原理（模仿學習 /Behavior Cloning）：先用遺傳演算法訓練一個神經網絡老師 →
  在評分賽道上跑老師、蒐集 `(狀態, 老師動作)` 資料集 → 用 CART 回歸樹擬合「狀態 → 老師的 flap 優勢」。
- 樹使用 sklearn 風格的平行陣列（`children_left/right/feature/threshold/value`），與 §1.2 完全一致。
- 單棵樹結構可解釋（能直接看到「dx 大於某值就拍翅膀」的規則）；森林靠多棵樹平均更穩。
- 兩種模式都能在評分賽道得到滿分（221）。

> 這兩條路徑讓學生對比「神經演化黑箱」vs「可解釋決策樹」vs「強化學習試錯」——
> 同一個任務、同一條評分賽道，分數卻由完全不同的方法得出，非常適合課堂討論。

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
