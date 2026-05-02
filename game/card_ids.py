"""カードID・カード名の定数定義。

各ファイルで文字列リテラルとして散在していたカードIDを集約。
タイポ防止・検索性向上・変更時の一括修正を目的とする。
"""

# ─── サポートカードID ───
RIRIE_NO_KESSHIN = "riirienokesshin"  # リーリエの決心
HAKASE_NO_KENKYU = "hakasenokenkyuu"  # 博士の研究
ZEIYU = "zeiyu"  # ゼイユ
HIKARI = "hikari"  # ヒカリ
JUDGE = "jixyajjiman"  # ジャッジマン
AKAMATSU = "akamatsu"  # アカマツ
MEI_NO_HAGEMASHI = "meinohagemashi"  # メイのはげまし
BOSS_NO_SHIREI = "bosunoshirei"  # ボスの指令
BURAIA = "buraia"  # ブライア
KIHADA = "kihada"  # キハダ
TANPAN_KOZOU = "tanpankozou"  # たんぱんこぞう

# ─── グッズカードID ───
HYPER_BALL = "haipaboru"  # ハイパーボール
POKEPAD = "pokepaddo"  # ポケパッド
FIGHT_GONG = "faitogongu"  # ファイトゴング
SUPER_BALL = "supaboru"  # スーパーボール
FUSHIGI_NA_AME = "fushiginaame"  # ふしぎなアメ
NAKAYOSHI_POFIN = "nakayoshipofuin"  # なかよしポフィン
YORU_NO_TANKA = "yorunotanka"  # 夜のタンカ
UNFAIR_STAMP = "anfeasutanpu"  # アンフェアスタンプ
SPECIAL_RED_CARD = "supeshiyarureddokado"  # スペシャルレッドカード
POKEMON_IREKAE = "pokemonirekae"  # ポケモンいれかえ
POKEMON_CATCHER = "pokemonkixyatchixya"  # ポケモンキャッチャー

# ─── どうぐカードID ───
FUUSEN = "fuusen"  # ふうせん
MAXIMUM_BELT = "makishimamuberuto"  # マキシマムベルト
POWER_PROTEIN = "pawapurotein"  # パワープロテイン

# ─── 手札刷新サポートID（リーリエ・博士等、手札を入れ替えるサポート） ───
HAND_REFRESH_SUPPORT_IDS = frozenset({
    RIRIE_NO_KESSHIN,
    HAKASE_NO_KENKYU,
    ZEIYU,
    HIKARI,
    JUDGE,
})

# ─── ボール系グッズID ───
BALL_GOODS_IDS = frozenset({
    FIGHT_GONG,
    POKEPAD,
    SUPER_BALL,
    HYPER_BALL,
    POKEMON_CATCHER,
    NAKAYOSHI_POFIN,
})
