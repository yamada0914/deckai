# 次ステップ：試合結果ベースの学習と現状整理

## 実装の安全な順番

1. **Self Play ログ収集**
2. **(state, action, result) データセット化**
3. **勝率モデル学習**
4. **evaluation function へ組み込み**

---

## 重要な改善ポイント（2 つ）

### ① state は「行動前」を保存する

- **❌ 悪い例**: `state_after_action` を保存する  
  → 「この状態は強い」は分かるが「どの行動が良かったか」が分からない。
- **✅ 正しい**: `state_before_action` + `action` + `result` を保存する。

つまり **`(state_t, action_t) → game_result`** の形にする。

### ② action を特徴量に入れる

logistic regression で **P(win | state, action)** を学習するなら、

- **features = state_features + action_features**
- action は **one-hot** にする。

例:

- **state**: サイド差、自分の HP 合計、相手の HP 合計、エネ数、ベンチ数、…
- **action**: `attach_target = lucario`, `attach_target = solrock` などを one-hot

---

## 次にやるべきこと（詳細）

### ③ attach と attack は別モデルで OK

- **model_attach** と **model_attack** に分けてよい（action space が違うため）。
- カードゲームでは一般的な設計。

### ④ minimax は「候補絞り」に使う

- **今**: `minimax → evaluation` で 1 手を選んでいる。
- **おすすめ**:  
  **minimax → 候補 3 手 → win probability model → 最終決定**

### ⑤ Self Play は最初ランダムを混ぜる（ε-greedy）

- **AI vs AI だけ**だと同じプレイばかり学習する。
- **例**: 90% best move / 10% random にするとデータの多様性がかなり良くなる。

### ⑥ 試合数

- ポケカは分岐が多いので **10000 〜 100000** 試合が目安。
- **同一デッキミラー**（例: Lucario deck mirror）なら **2〜3 万**で十分な可能性あり。

### ⑦ 完全情報は今は正解

- 最初は **perfect information** で作るのが普通（学習が安定する。AlphaZero も同様）。
- あとで **belief state** に拡張できる。

---

## 相手手札の扱い

**答え: 1️⃣ 完全情報（見えている）**

- シミュレーションでは両プレイヤーの手札・山札・場をすべて参照している。
- 実戦（人間対戦）で部分観測にする場合は、のちに belief state 等で拡張する。

---

## 現状の壁の整理

| 要素         | 現状                     | 問題                         |
|--------------|--------------------------|------------------------------|
| 技選択       | minimax が強い           | 重みの寄与がほぼ出ない       |
| エネルギー付与 | 重み学習すると 40% に悪化 | 学習指標が間違っている       |
| 学習指標     | 手の直後の盤面評価       | 長期効果を無視（クレジットアサインメント） |

---

## 次に来る「ほぼ確実な壁」：評価関数が壊れる問題

- ポケカは **サイド 6 → 0** なので、**途中盤面の強さ**が非常に分かりにくい。
- 対策として **intermediate reward** を少し入れることがある。  
  例: サイド 1 枚 +0.2、ポケモンきぜつ +0.1  
- **入れすぎると「ズルい戦略」を覚える**ので、控えめにする。

---

## 次の改善の優先順位（推奨）

1. **① attach を top2 分岐する（最優先）**  
   現状は run_turn_auto → attach 1 択（貪欲）なので、attach ミスがそのまま確定する。ポケカでは「attach A → 勝ち / attach B → 負け」の分岐が多い。**attach candidates = top2** にし、attack × attach で branching ≈ 2×2 = 4 に抑える。計算量はほぼ増えない。

2. **② サーチの deterministic はこのままでよい**  
   サーチ結果を random にすると探索ノイズが増える。AlphaZero 系でも decision = deterministic / chance = random の分離が一般的。現状（shuffle=random, coin=random, search choice=deterministic）は合理的。

3. **③ コインは今は random のままでよい**  
   平均化（期待値）を入れると expectimax になるが計算量が増える。現段階では random sampling の方が扱いやすい。

4. **④ Self Play ログを作り始める**  
   探索はすでに安定しているので、学習データを作る段階。ログ形式は **state_before_action**, action, player, game_result。**state_before_action** を必ず保存する。

5. **⑤ attach の特徴量**  
   attach_target に加えて **next_turn_damage**, **energy_missing**, **is_active**, **hp** を入れると強い。  
   例: `features = [side_diff, active_hp, bench_hp, energy_on_target, energy_needed, next_turn_damage]` など。

6. **⑥ シミュレーション時間を必ず測る**  
   理想は 1 game < 0.1 秒。1 game = 0.5 秒なら 50k games ≈ 7 時間。

**推奨実装順**: ① attach top2 分岐 → ② Self play 30k → ③ attach policy 学習 → ④ evaluation learning。この順でかなり強くなる可能性が高い。

---

## 現在の AI の構造

**今**:
```
Rule Engine
  ↓
run_turn_auto (greedy, 固定順序)
  ↓
minimax (attack branching only)
  ↓
evaluation（重み）
```

**次の進化**:
```
Rule Engine
  ↓
learned attach（top2 分岐 or policy）
  ↓
minimax
  ↓
learned evaluation（P(win | state, action)）
```

おすすめの将来形:

```
minimax
  ↓ 候補 3 手に絞る
win probability model (P(win | state, action))
  ↓ 最終決定
```

---

## 技術メモ（要更新）

| 項目 | 現状 |
|------|------|
| **使用言語** | Python |
| **minimax depth** | **2**（実質 2.5 手読み: 自分攻撃 → 相手ターン → 自分ターン run_turn_auto → 盤面評価） |
| **1 試合シミュレーション時間** | 要計測（理想は 1 試合 &lt; 0.1 秒。1 秒だと 100k 試合 ≒ 27 時間） |

---

## run_turn_auto の範囲（分岐の話）

**結論: サポート・グッズ・進化・にげる・エネルギー付与・攻撃を「全部やっている」が、探索はしていない（1 ターン 1 本の貪欲パス）。**

### やっていること（順序は固定）

`turn.py` の `run_turn_auto` は次の順で**可能なものを貪欲に 1 つずつ実行**する:

0. ベンチにポケモンを出す（空きがある限り）
1. 進化（ふしぎなアメ含む・複数回）
2. スタジアム（1 枚まで）
3. 手札を捨てないサポート（キハダ・ネモ）
4. エネルギー付与（1 枚、重みで付与先選択）
5. ボール系グッズ（キャッチャー優先、重みで対象選択）
6. サポート（博士の研究等・重みで使用順）
7. いれかえ（生存用）
8. にげる（KO 確定時・自発的・重みで逃げ先選択）
9. どうぐ・グッズ（重みで使用順）
10. 進化（再）
11. 攻撃（minimax + 重みで技選択）

サポート／グッズを 1 回使うと、先頭の進化から再度チェック（`MAX_TURN_ACTION_ROUNDS` 回まで）。

### 分岐（branching factor）について

- **run_turn_auto 内**: 各カテゴリで「決まった順序／重みでソートした候補」の**先頭 1 つ**を実行するだけ。**ツリー探索はしていない**。1 ターンにつき **1 本のパス** のみ。
- **minimax で分岐しているのは攻撃だけ**: 合法攻撃それぞれに対して `run_turn_auto` をシミュレートしている。なので **branching factor ≒ 合法攻撃の数**（多くても 1〜2 のことが多い）。

つまり「サポート・グッズ・進化・にげるを全部探索しているか？」→ **全部やっているが、探索ではなく貪欲 1 手**。攻撃だけが minimax で分岐している。

**評価**: branching ≒ 攻撃数（多くても 2 程度）なので探索は安定している。全行動を探索すると branching ≈ 100〜1000 になりがちで、ここで破綻するポケカ AI は多い。

---

## 次にやると効果が大きい改善（重要順）

1. **attach の候補だけ少し探索する**  
   エネルギー付与もゲームを大きく左右するので、**attach の top2 だけ分岐**する。  
   構造例: 攻撃 2 通り × attach 2 通り → branching 4 程度でまだ軽い。

2. **attack candidate を top3 にする**  
   minimax でスコアを出し、**top3 だけ残す** → **P(win \| state, action)** で最終決定。

3. **attach policy を学習**  
   `(state_before_attach, attach_target, result)` で学習。logistic regression または small MLP。

4. **attack policy も同様**  
   特徴量例: damage, ko, energy_cost, hp_remaining など。

**方針**: `run_turn_auto` はルールベース AI。ここに **learned policy** を少しずつ入れていく。  
rule AI → rule + learned attach → rule + learned attach + learned attack。  
さらに Self Play で (state, action, win) データセットを作り、P(win \| state, action) を学習して evaluation に置き換えるとかなり強くなる。最終形は **policy network + minimax**（minimax → candidate actions → policy prior → selection、AlphaZero と同じ思想）。

---

## Self Play と state encoding の設計メモ

### SelfPlayRunner のイメージ

```text
for i in range(50000):
    game = setup_game(deck, deck)
    log = []
    while not game.is_terminal():
        action = ai.choose_action(game, epsilon=0.1)
        log.append({
            "state": encode_state(game),   # 行動前の state
            "action": action,
            "player": game.current_player
        })
        game.step(action)
    result = game.winner
    for record in log:
        record["result"] = 1 if record["player"] == result else 0
```

→ **(state_before_action, action, result)** のデータが得られる。

### state encoding は先に決める（20〜50 次元程度）

- 最初はシンプルに。例: 自分・相手それぞれ「サイド枚数、アクティブ HP、ベンチ HP 合計、エネルギー数、ベンチ数」など。
- カード ID を全部入れると 1000 次元超えて学習が壊れやすいので、最初は控える。

### attach の特徴量

- **attach_target** に加えて **future_attack_damage** 系を入れると強い。
- 例: `damage_next_turn`, `energy_needed`, `is_active` など。

### attach 学習のログは attack と混ぜない

- 保存するのは **state_before_attach** と **attach_target**。
- attack 用ログとは別にしておく。

### logistic regression と minimax 統合

- **P(win \| state, action)** がそのまま evaluation になる: `eval = sigmoid(w·x)`。
- おすすめ: **minimax → 候補 top3 → P(win \| state, action) で max → 最終決定**。

---

## おすすめの具体的ステップ

1. **Self Play ログ収集**（ε-greedy で 90% best / 10% random）。目安 2万〜10万試合（ミラーなら 2〜3 万でも可）。
2. **ログ形式**: 各手について **state_before_action**（またはその特徴量）、**action**（one-hot 用）、**player**, **turn**。試合終了後に **game_result** を全手に付与。
3. **特徴量**: `state_features` + `action_features`（one-hot）。**P(win \| state, action)** を logistic regression で学習。
4. **model_attach** と **model_attack** は別々に学習。
5. **evaluation / 意思決定へ組み込み**: minimax で候補を数手に絞り、その候補に対して win probability model でスコアを付け、最終決定。

この構成（minimax + action 分離 + self play + 勝敗学習への移行）は研究レベルでもよく使われる。

---

## 確率イベントの処理（山札サーチ・手札交換・シャッフル・コイン）

**結論: ほぼ ① ランダム。サーチで「取る1枚」を決めるときだけ ② deterministic。③ 平均化は未使用。**

| 種類 | 実装 | 分類 |
|------|------|------|
| **山札シャッフル** | `random.shuffle(deck)`（初手・引き直し・博士の研究・たんぱんこぞう・キハダ・ふきあらす等） | ① ランダム |
| **ドロー** | `deck.pop()`（山札の上から）。順序は直前のシャッフルで決まる | ① ランダム（シャッフルに依存） |
| **コイン** | `_flip_coin()` → `random.random() < 0.5`（ねむり解除・こんらん・マグネリジェクト等） | ① ランダム |
| **相手ベンチから1体選ぶ** | `random.randint(0, len(opp.bench)-1)`（ボスの指令・マグネリジェクト・ふきあらす等） | ① ランダム |
| **先行／後攻** | `random.randint(0, 1)` | ① ランダム |
| **山札サーチで「取る1枚」** | ポケパッド: 重みで `max(..., key=score)`。トモダチをさがす: 山札先頭から最初のポケモン。取った後に `random.shuffle(deck)` | ② deterministic（選び方）＋ ① ランダム（シャッシュ後の山札順） |
| **平均化（期待値で評価）** | 未実装 | ③ なし |

シミュレーションの再現性を出したい場合は `setup_game(seed=...)` で固定可能。minimax 内の `copy_for_simulation` では seed は引き継がれず、各シミュレーションで別のランダム列になる。

---

## 上級テクニック: action ordering

今の `run_turn_auto` は**固定順序**（ベンチ → 進化 → サポート → エネ → グッズ …）で、プレイが制限される。  
例: **ボール → 進化** の順が強い局面がある。

上級 AI では **action ordering を少し探索**するが、**branching が爆発しやすい**ので慎重にやる。  
（例: 順序の候補を 2〜3 パターンに絞る、または学習で「どの順で試すか」を決めるなど。）
