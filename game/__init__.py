"""
ポケカ風ルールのゲームロジック。

- 初手 7 枚、ターンでドロー・エネルギー付与・アイテム・攻撃。
- ベンチは最大 5 体。きぜつしたらベンチから 1 体をバトル場に出す。
- 相手のポケモンを 6 回きぜつさせたら勝ち。プレイヤーは先行・後攻の 2 人。
"""
from .state import (
    BENCH_SIZE,
    WIN_KO_COUNT,
    PRIZE_COUNT,
    MAX_TURNS_SAFETY,
    MAX_EVOLVE_ROUNDS_PER_TURN,
    MAX_BALL_USES_PER_TURN,
    MAX_TURN_ACTION_ROUNDS,
    BattlePokemon,
    GameState,
    PlayerState,
    setup_game,
    start_turn,
    _check_game_end,
    rules_only_for_player,
)
from .weights import GameWeights, load_weights, save_weights
from .evolution import evolve_pokemon
from .trainers import (
    attach_energy,
    use_potion,
    use_trainer_goods,
    attach_tool,
    use_pokemon_swap,
    use_support,
)
from .attack import attack
from .damage import _effective_damage_to_defender
from .evaluate import evaluate_board
from .turn import retreat, end_turn, run_turn_auto, run_game_auto
from .policy_rules_only import pick_energy_attach_candidate

__all__ = [
    "BENCH_SIZE",
    "WIN_KO_COUNT",
    "PRIZE_COUNT",
    "MAX_TURNS_SAFETY",
    "BattlePokemon",
    "GameState",
    "GameWeights",
    "PlayerState",
    "setup_game",
    "start_turn",
    "evolve_pokemon",
    "attach_energy",
    "retreat",
    "use_support",
    "use_potion",
    "use_trainer_goods",
    "attack",
    "run_turn_auto",
    "run_game_auto",
    "end_turn",
    "evaluate_board",
    "_check_game_end",
    "_effective_damage_to_defender",
    "load_weights",
    "save_weights",
    "pick_energy_attach_candidate",
    "rules_only_for_player",
]
