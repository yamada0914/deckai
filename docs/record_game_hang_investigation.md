# record_game.py が終わらない原因調査

## 概要

`python scripts/record_game.py`（または `python record_game.py`）が終了しない場合に考えられる原因を整理した。

---

## 1. メインループの流れ（scripts/record_game.py）

```python
while True:
    start_turn(state)           # ドロー1枚、ねむり解除など
    if state.winner is not None:
        break
    run_turn_auto(state)        # 1ターン分の行動をすべて実行
    if _check_game_end(state):  # サイド0 or ポケモン全滅で勝敗判定
        break
    end_turn(state)             # ターン交代、turn_count += 1
    if state.turn_count >= 200: # 安全打ち切り
        break
```

- ゲームが終わる条件: **誰かのサイドが0** / **誰かのバトル場・ベンチが両方0** / **デッキ切れでドロー不可** / **200ターンで打ち切り**
- 終わらない = 上記のどれにも到達せず、かつ「ある処理」が戻ってこないか、極端に遅いと考えられる。

---

## 2. 想定される原因

### A. `run_turn_auto(state)` が戻らない（無限ループ）

`game.run_turn_auto` 内の **while True** は次の2つだけ。

| 場所 | 内容 | 抜け方 |
|------|------|--------|
| 進化ブロック（2箇所） | `while can_evolve:` | 1回でも進化しなければ `evolved_this_round` が False のまま `break` |
| ボール系グッズ | `while True:` … ボールを1枚使う or 使えなければ `break` | 手札のボールが減る or 使えるボールがなくなるので有限回で終了 |

- ロジック上はどちらも「有限回で抜ける」想定。
- **ただし** どこかで「1回の use_trainer_goods / evolve_pokemon で手札・山札が同じ形に戻る」ようなバグや、条件判定ミスがあると、同じ状態を繰り返して実質無限ループになる可能性はある。

### B. `run_turn_auto` は戻るが、1ターンあたりの処理が非常に重い

- `run_turn_auto` のあちこちで **`state._record_frame()`** が呼ばれる。
- `record_frame(s)` は **`copy.deepcopy(s)`** で状態全体をコピーし、`states` と `log_snapshots` に追加している。
- 1ターンで「進化・ボール・いれかえ・にげる・どうぐ・グッズ・攻撃」と多数の行動をすると、**1ターンあたり数十回以上 _record_frame が呼ばれる** ことがある。
- そのたびに `copy.deepcopy(state)` が走るため、
  - 状態が大きい（デッキ・手札・捨て札が多い）
  - または `state` に循環参照などが入っている  
  だと **1ターンが極端に遅く** なり、「終わらない」ように見える可能性がある。

### C. `copy.deepcopy(state)` が遅い or 止まる

- `GameState` は `log_fn` / `record_frame_fn` にクロージャを渡している。
- 通常、`deepcopy` は関数オブジェクトをそのまま参照でコピーするだけなので、クロージャ経由で `state` 自身を参照していない限り、循環参照にはなりにくい。
- ただし **デッキ・手札・捨て札・ベンチなどのリストが大きい** と、`deepcopy` のコストだけで 1 フレームあたりが重くなり、全体として「終わらない」ように感じることはあり得る。

### D. ゲームが終了条件を満たさない（200ターンまで進む）

- 勝敗がつかずに **毎ターン end_turn が呼ばれ、turn_count が 200 まで増える** と、そこで打ち切りになる。
- 「終わらない」のが「200ターン待っている間ずっと動いている」のであれば、**終了条件（サイド0 / ポケモン0 / デッキ切れ）に到達しにくいデッキ構成や AI の行動** が原因の可能性がある。
- その場合、ループ自体は止まらず、**200ターン到達まで時間がかかりすぎている** という意味の「終わらない」になり得る。

---

## 3. 切り分けのための確認方法（修正はしない）

1. **どこで止まっているか見る**
   - `scripts/record_game.py` の `while True:` の直下に  
     `print(f"turn_count={state.turn_count}")` を入れて実行。
   - 同じ `turn_count` が何度も出る → そのターンの **`run_turn_auto` 内** で時間がかかっているか、戻っていない。
   - `turn_count` が少しずつ増える → ループは進んでいるが **1ターンあたりの処理（主に deepcopy）が重い** 可能性が高い。

2. **run_turn_auto の入口でログ**
   - `game.run_turn_auto` の先頭に  
     `if state.log_fn: state.log(f"[DEBUG] run_turn_auto start turn_count={state.turn_count}")` を入れて実行。
   - 同じ turn_count の DEBUG が何度も出る → そのターンで **run_turn_auto が複数回呼ばれている** のではなく（record_game では1ターン1回）、**その1回の run_turn_auto のなかで _record_frame が大量に呼ばれている** か、内部のどこかでループしている。

3. **フレーム数で重さを確認**
   - 上記の `print` の代わりに、`record_frame` 内で `print(len(states))` を実行。
   - 数が一気に増え続ける → 1ターンで **_record_frame が非常に多く呼ばれている**。deepcopy の負荷が「終わらない」原因の候補。

4. **200ターン打ち切りまで待つ**
   - 長時間放置して、最終的に「打ち切り」のログとともに終了するか確認。
   - 終了する → 無限ループではなく、**試合が長い or 1ターンあたりの処理が重い** ことが原因。

---

## 4. 結論（原因の優先度）

| 優先度 | 想定原因 | 確認方法 |
|--------|----------|----------|
| 高 | 1ターンあたりの **`_record_frame` 呼び出し回数が多く、`copy.deepcopy(state)` の負荷で極端に遅い** | 上記 1 で turn_count が増えているか、3 で states の増え方を見る |
| 中 | **`run_turn_auto` 内のどこかで実質的な無限ループ**（ボール・進化の条件ミスなど） | 上記 1 で同じ turn_count が延々と出るか確認 |
| 低 | ゲームが長引き、**200ターン打ち切りまで時間がかかっている** | 4 で打ち切りまで待つ |

修正は行わず、上記の切り分けをしたうえで、必要なら「フレーム間引き」「deepcopy の軽量化」「ループガード」などの対策を検討するとよい。
