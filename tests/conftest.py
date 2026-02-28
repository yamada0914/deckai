"""
pytest 用 conftest。リポジトリルートを path に追加し、ルールベーステスト用の共通フィクスチャを提供する。

参照ルール:
- rules/01_play_supplement.md  （遊びかた説明書の補足）
- rules/02_card_descriptions.md （カードの説明文）
- rules/advanced_rule.md       （上級プレイヤー用ルールガイド）
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
