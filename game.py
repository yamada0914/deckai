"""
ポケカ風ルールのゲームロジック。

- 初手 4 枚、ターンでドロー・エネルギー付与・アイテム・攻撃。
- ベンチは最大 5 体。きぜつしたらベンチから 1 体をバトル場に出す。
- 相手のポケモンを 3 回きぜつさせたら勝ち。プレイヤーは先行・後攻の 2 人。
"""
import random
from dataclasses import dataclass, field
from typing import Callable, Literal

from card import PokemonCard, is_energy, is_item, is_pokemon
from deck import STARTING_HAND_SIZE, create_deck, format_deck_recipe

BENCH_SIZE = 5
WIN_KO_COUNT = 3
MAX_TURNS = 200


@dataclass
class BattlePokemon:
    """バトル場／ベンチに出すポケモン（HP と付いているエネルギーを管理）"""
    card: PokemonCard
    attached_energy: int = 0

    @property
    def hp(self) -> int:
        return self.card.hp

    @hp.setter
    def hp(self, value: int) -> None:
        self.card.hp = min(value, self.card.max_hp)

    @property
    def max_hp(self) -> int:
        return self.card.max_hp


@dataclass
class PlayerState:
    deck: list = field(default_factory=list)
    hand: list = field(default_factory=list)
    active: BattlePokemon | None = None
    bench: list[BattlePokemon] = field(default_factory=list)
    knockouts_suffered: int = 0

    def draw(self, n: int = 1) -> list:
        drawn = []
        for _ in range(n):
            if not self.deck:
                return drawn
            drawn.append(self.deck.pop())
        return drawn

    def has_active_pokemon(self) -> bool:
        return self.active is not None and self.active.hp > 0


def _card_label(card) -> str:
    return getattr(card, "name", card.__class__.__name__)


def _player_name(state: "GameState", i: int) -> str:
    """使用デッキに応じたプレイヤー名を返す。同じデッキ同士のときはデッキ1/デッキ2、それ以外はオタチ/ワニ/カエルデッキ。"""
    d0, d1 = state.deck_indices[0], state.deck_indices[1]
    if d0 == d1:
        return f"デッキ{i + 1}"
    names = ["オタチデッキ", "ワニデッキ", "カエルデッキ"]
    idx = state.deck_indices[i] if i < len(state.deck_indices) else i
    return names[idx] if idx < len(names) else f"プレイヤー{i}"


def _turn_label(state: "GameState") -> str:
    """ログ用に「先行Nターン」「後行Nターン」のラベルを返す。"""
    is_first_player_turn = state.current_player == state.first_player
    n = (state.turn_count // 2) + 1
    return f"先行{n}ターン" if is_first_player_turn else f"後行{n}ターン"


@dataclass
class GameState:
    current_player: Literal[0, 1] = 0
    players: list[PlayerState] = field(default_factory=lambda: [PlayerState(), PlayerState()])
    turn_count: int = 0
    first_player: Literal[0, 1] = 0
    winner: Literal[0, 1, -1] | None = None
    log_fn: Callable[[str], None] | None = None
    deck_indices: tuple = (0, 1)
    shippegaeshi_120_used: bool = False

    def log(self, msg: str) -> None:
        if self.log_fn:
            self.log_fn(msg)

    def opponent(self) -> int:
        return 1 - self.current_player

    def active_player_state(self) -> PlayerState:
        return self.players[self.current_player]

    def defending_player_state(self) -> PlayerState:
        return self.players[self.opponent()]

    def player_name(self, i: int) -> str:
        return _player_name(self, i)


def setup_game(
    seed: int | None = None,
    log_fn: Callable[[str], None] | None = None,
    deck0: int = 0,
    deck1: int = 1,
) -> GameState:
    """デッキを組んで初手 4 枚をセット。deck0 / deck1 で使用デッキを指定（0=A, 1=B, 2=C）。"""
    if seed is not None:
        random.seed(seed)
    state = GameState(log_fn=log_fn, deck_indices=(deck0, deck1))
    state.log("========== ゲーム開始 ==========")
    for i in range(2):
        state.log(f"{state.player_name(i)} デッキ: [{format_deck_recipe(state.deck_indices[i])}]")
    for i in range(2):
        state.players[i].deck = create_deck(state.deck_indices[i])
        random.shuffle(state.players[i].deck)
        state.log(f"{state.player_name(i)}: デッキ {len(state.players[i].deck)} 枚をシャッフル")
        retry_count = 0
        while True:
            state.players[i].hand = state.players[i].draw(STARTING_HAND_SIZE)
            hand_names = [_card_label(c) for c in state.players[i].hand]
            has_basic_pokemon = any(
                is_pokemon(c) and not getattr(c, "evolves_from", None) for c in state.players[i].hand
            )
            if has_basic_pokemon:
                break
            retry_count += 1
            state.players[i].deck.extend(state.players[i].hand)
            state.players[i].hand = []
            random.shuffle(state.players[i].deck)
            state.log(f"{state.player_name(i)}: 初手にたねポケモンなし → 再シャッフルして引き直し")
        msg = f"再シャッフル後、初手 4 枚ドロー → 手札 [{', '.join(hand_names)}]" if retry_count > 0 else f"初手 4 枚ドロー → 手札 [{', '.join(hand_names)}]"
        state.log(f"{state.player_name(i)}: {msg}")
        _put_one_pokemon_active(state.players[i])
        if state.players[i].active:
            state.log(f"{state.player_name(i)}: バトル場に {state.players[i].active.card.name} を出す（HP {state.players[i].active.max_hp}）")
        _fill_bench_from_hand(state.players[i], state, i, log_each=False)
        if state.players[i].bench:
            bench_names = [b.card.name for b in state.players[i].bench]
            state.log(f"{state.player_name(i)}: ベンチに {len(state.players[i].bench)} 体 [{', '.join(bench_names)}]")
    state.first_player = random.randint(0, 1)
    state.current_player = state.first_player
    state.log(f"先行: {state.player_name(state.first_player)} / 後攻: {state.player_name(1 - state.first_player)}")
    state.log("")
    return state


def _put_one_pokemon_active(player: PlayerState) -> bool:
    """手札からたねポケモン（進化ポケモンでない）1 体をバトル場に出す。出せなければ何もしない。"""
    for i, c in enumerate(player.hand):
        if is_pokemon(c) and not c.evolves_from:
            p = c.copy()
            player.active = BattlePokemon(card=p)
            player.hand.pop(i)
            return True
    return False


def _put_one_pokemon_on_bench(
    player: PlayerState, state: GameState, player_index: int, *, log: bool = True
) -> bool:
    """手札からたねポケモン（進化ポケモンでない）1 体をベンチに出す（ベンチが 5 体未満のとき）。"""
    if len(player.bench) >= BENCH_SIZE:
        return False
    for i, c in enumerate(player.hand):
        if is_pokemon(c) and not c.evolves_from:
            p = c.copy()
            player.bench.append(BattlePokemon(card=p))
            player.hand.pop(i)
            if log:
                state.log(f"{state.player_name(player_index)}: ベンチに {p.name} を出す（ベンチ {len(player.bench)} 体）")
            return True
    return False


def _fill_bench_from_hand(
    player: PlayerState, state: GameState, player_index: int, *, log_each: bool = True
) -> None:
    """手札のポケモンをベンチに出す（最大 BENCH_SIZE まで）。"""
    while len(player.bench) < BENCH_SIZE and _put_one_pokemon_on_bench(
        player, state, player_index, log=log_each
    ):
        pass


def _promote_from_bench(player: PlayerState, state: GameState, player_index: int) -> bool:
    """バトル場が空のとき、ベンチから 1 体をバトル場に出す。"""
    if player.active is not None or not player.bench:
        return False
    player.active = player.bench.pop(0)
    state.log(f"{state.player_name(player_index)}: ベンチから {player.active.card.name} をバトル場に出す（HP {player.active.max_hp}）")
    return True


def _handle_opponent_ko(opp: PlayerState, state: GameState) -> bool:
    """相手のきぜつを 1 回加算し、勝ちなら state.winner をセットして True、否則ベンチから繰り出して False。"""
    opp.knockouts_suffered += 1
    if opp.knockouts_suffered >= WIN_KO_COUNT:
        state.winner = state.current_player
        return True
    _promote_from_bench(opp, state, state.opponent())
    return False


def _check_game_end(state: GameState) -> bool:
    """勝敗が決まっていれば state.winner をセットして True を返す。3 回きぜつで負け、またはバトル場・ベンチともにいないと負け。"""
    for i in range(2):
        p = state.players[i]
        if p.knockouts_suffered >= WIN_KO_COUNT:
            state.winner = 1 - i
            state.log(f"{state.player_name(i)} のポケモンが {WIN_KO_COUNT} 回きぜつ → {state.player_name(state.winner)} の勝ち")
            return True
        if not p.has_active_pokemon() and len(p.bench) == 0:
            state.winner = 1 - i
            state.log(f"{state.player_name(i)} のバトル場・ベンチにポケモンがいない → {state.player_name(state.winner)} の勝ち")
            return True
    return False


def start_turn(state: GameState) -> None:
    """ターン開始：ドロー 1 枚（先行もドローあり）。ドローできなければデッキ切れ負け。"""
    p = state.active_player_state()
    turn_label = _turn_label(state)
    state.log(f"---------- {turn_label} ---------- {state.player_name(state.current_player)} のターン")
    drawn = p.draw(1)
    if drawn:
        p.hand.extend(drawn)
        state.log(f"{state.player_name(state.current_player)}: 1 枚ドロー → {_card_label(drawn[0])}（手札 {len(p.hand)} 枚、デッキ {len(p.deck)} 枚）")
    else:
        state.log(f"{state.player_name(state.current_player)}: デッキが空のためドローなし（手札 {len(p.hand)} 枚）")


def _opponent_max_damage(state: GameState) -> int:
    """相手のバトル場ポケモンが今出せる最大ダメージを返す（エネルギーに応じた最強ワザ）。"""
    opp = state.defending_player_state()
    if not opp.active or not opp.active.card.attacks:
        return 0
    max_dmg = 0
    for atk in opp.active.card.attacks:
        if opp.active.attached_energy >= atk.energy_cost:
            dmg = atk.damage
            if atk.name == "しっぺがえし" and state.active_player_state().knockouts_suffered == 2:
                dmg += 90
            max_dmg = max(max_dmg, dmg)
    return max_dmg


def _our_max_damage(state: GameState) -> int:
    """自分のバトル場ポケモンがこのターン出せる最大ダメージを返す。"""
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not p.active or not p.active.card.attacks:
        return 0
    max_dmg = 0
    for atk in p.active.card.attacks:
        if p.active.attached_energy >= atk.energy_cost:
            dmg = atk.damage
            if atk.name == "しっぺがえし" and opp.knockouts_suffered == 2:
                dmg += 90
            max_dmg = max(max_dmg, dmg)
    return max_dmg


def retreat(state: GameState, bench_index: int) -> bool:
    """バトル場のポケモンとベンチの bench_index 番目を入れ替える（逃げる）。にげるエネルギー分を捨てる。"""
    p = state.active_player_state()
    if not p.active or bench_index < 0 or bench_index >= len(p.bench):
        return False
    old_active = p.active
    cost = getattr(old_active.card, "retreat_cost", 1)
    if old_active.attached_energy < cost:
        return False
    old_active.attached_energy -= cost
    if cost > 0:
        state.log(
            f"{state.player_name(state.current_player)}: 逃げるために {old_active.card.name} のエネルギーを {cost} 個捨てる"
        )
    p.active = p.bench[bench_index]
    p.bench[bench_index] = old_active
    state.log(
        f"{state.player_name(state.current_player)}: {old_active.card.name} をベンチに退き、"
        f"{p.active.card.name} をバトル場に出す（HP {p.active.hp}/{p.active.max_hp}）"
    )
    return True


def attach_energy(state: GameState, hand_index: int, bench_index: int | None = None) -> bool:
    """手札の energy_index 番目のエネルギーをバトル場またはベンチのポケモンに付与。bench_index 指定でベンチに付与。"""
    p = state.active_player_state()
    if hand_index < 0 or hand_index >= len(p.hand):
        return False
    if not is_energy(p.hand[hand_index]):
        return False
    if bench_index is not None:
        if bench_index < 0 or bench_index >= len(p.bench):
            return False
        target = p.bench[bench_index]
        target.attached_energy += 1
        p.hand.pop(hand_index)
        state.log(
            f"{state.player_name(state.current_player)}: エネルギーを 1 つ付与（ベンチの {target.card.name}、エネルギー {target.attached_energy} 個）"
        )
        return True
    if not p.active:
        return False
    p.active.attached_energy += 1
    p.hand.pop(hand_index)
    state.log(f"{state.player_name(state.current_player)}: エネルギーを 1 つ付与（バトル場のエネルギー {p.active.attached_energy} 個）")
    return True


def use_potion(state: GameState, hand_index: int) -> bool:
    """きずぐすりを使用して自分のバトル場のポケモンを 20 回復。"""
    p = state.active_player_state()
    if not p.active or hand_index < 0 or hand_index >= len(p.hand):
        return False
    card = p.hand[hand_index]
    if not is_item(card) or getattr(card, "effect", None) != "heal":
        return False
    amount = getattr(card, "heal_amount", 20)
    before = p.active.hp
    p.active.hp = min(p.active.hp + amount, p.active.max_hp)
    p.hand.pop(hand_index)
    state.log(f"{state.player_name(state.current_player)}: きずぐすりを使用 → バトル場のポケモンを {before} → {p.active.hp} に回復")
    return True


def _apply_evolution(
    target: BattlePokemon,
    evolution_card: PokemonCard,
    state: GameState,
    log_prefix: str,
) -> None:
    """進化を 1 体に適用（HP・エネルギーを引き継ぎ、カード差し替え）。"""
    old_name = target.card.name
    old_hp = target.hp
    old_max_hp = target.card.max_hp
    old_energy = target.attached_energy
    damage_taken = old_max_hp - old_hp
    evolved = evolution_card.copy()
    target.card = evolved
    target.card.max_hp = evolved.max_hp
    target.card.hp = max(0, min(evolved.max_hp, evolved.max_hp - damage_taken))
    target.attached_energy = old_energy
    state.log(f"{log_prefix}{old_name} を {evolved.name} に進化（HP {target.hp}/{target.max_hp}）")


def evolve_pokemon(state: GameState, hand_index: int, bench_index: int | None = None) -> bool:
    """
    手札の進化ポケモンで、バトル場またはベンチのポケモンを進化させる。
    bench_index=None ならバトル場、数値ならベンチのそのインデックス。
    """
    p = state.active_player_state()
    if hand_index < 0 or hand_index >= len(p.hand):
        return False
    evolution_card = p.hand[hand_index]
    if not is_pokemon(evolution_card) or not evolution_card.evolves_from:
        return False

    if bench_index is None:
        if not p.active or p.active.card.id != evolution_card.evolves_from:
            return False
        _apply_evolution(
            p.active, evolution_card, state,
            f"{state.player_name(state.current_player)}: ",
        )
        p.hand.pop(hand_index)
        return True
    if bench_index < 0 or bench_index >= len(p.bench):
        return False
    bench_pokemon = p.bench[bench_index]
    if bench_pokemon.card.id != evolution_card.evolves_from:
        return False
    _apply_evolution(
        bench_pokemon, evolution_card, state,
        f"{state.player_name(state.current_player)}: ベンチの ",
    )
    p.hand.pop(hand_index)
    return True


def attack(
    state: GameState,
    attack_index: int,
) -> bool:
    """
    攻撃を実行。
    相手にダメージ、自分に self_damage があれば自分にもダメージ。
    きぜつしたらバトル場から外れ、勝敗判定になる。
    """
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not p.active or not opp.active:
        return False
    pokemon_card = p.active.card
    if attack_index < 0 or attack_index >= len(pokemon_card.attacks):
        return False
    atk = pokemon_card.attacks[attack_index]
    if p.active.attached_energy < atk.energy_cost:
        return False
    opp_before = opp.active.hp
    damage = atk.damage
    if atk.name == "しっぺがえし" and opp.knockouts_suffered == 2:
        damage += 90
        state.shippegaeshi_120_used = True
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」で相手に {damage} ダメージ（相手あと1きぜつで負けのため 90 ダメージ追加、相手 HP {opp_before} → {opp_before - damage}）")
    else:
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」で相手に {damage} ダメージ（相手 HP {opp_before} → {opp_before - damage}）")
    opp.active.hp -= damage
    bench_dmg = getattr(atk, "bench_damage", 0)
    if bench_dmg > 0 and opp.bench:
        bench_target = opp.bench[0]
        bench_before = bench_target.hp
        bench_target.hp -= bench_dmg
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」で相手のベンチの {bench_target.card.name} に {bench_dmg} ダメージ（HP {bench_before} → {bench_target.hp}）")
        if bench_target.hp <= 0:
            state.log(f"{state.player_name(state.opponent())} のベンチの {bench_target.card.name} がきぜつ！（{opp.knockouts_suffered + 1} 回目）")
            opp.bench.pop(0)
            if _handle_opponent_ko(opp, state):
                return True

    self_dmg = getattr(atk, "self_damage", 0)
    if self_dmg > 0 and p.active:
        self_before = p.active.hp
        p.active.hp -= self_dmg
        state.log(f"{state.player_name(state.current_player)}: 反動で自分に {self_dmg} ダメージ（自分 HP {self_before} → {p.active.hp}）")

    if opp.active and opp.active.hp <= 0:
        state.log(f"{state.player_name(state.opponent())} のバトル場のポケモンがきぜつ！（{opp.knockouts_suffered + 1} 回目）")
        opp.active = None
        if _handle_opponent_ko(opp, state):
            return True

    if p.active and p.active.hp <= 0:
        state.log(f"{state.player_name(state.current_player)} のバトル場のポケモンが反動できぜつ！")
        p.active = None
        _promote_from_bench(p, state, state.current_player)
    return True


def _choose_best_attack_index(state: GameState, p: PlayerState, opp: PlayerState) -> int | None:
    """出せる技のうちダメージが最大のインデックスを返す。出せなければ None。"""
    if not p.active or not p.active.card.attacks:
        return None
    best_idx = None
    best_dmg = -1
    for idx, atk in enumerate(p.active.card.attacks):
        if p.active.attached_energy < atk.energy_cost:
            continue
        dmg = atk.damage
        if atk.name == "しっぺがえし" and opp.knockouts_suffered == 2:
            dmg += 90
        if dmg > best_dmg:
            best_dmg = dmg
            best_idx = idx
    return best_idx


def _should_attach_for_evolution(p: PlayerState) -> bool:
    """バトル場のポケモンが手札の進化で必要エネルギーに足りていなければ True。"""
    if not p.active:
        return False
    for c in p.hand:
        if not (is_pokemon(c) and getattr(c, "evolves_from", None) == p.active.card.id):
            continue
        evolved_max_cost = max((a.energy_cost for a in c.attacks), default=0)
        return p.active.attached_energy < evolved_max_cost
    return False


def _energy_needed_for_active(p: PlayerState) -> int:
    """バトル場（＋手札の進化）に必要なエネルギーコストを返す。"""
    if not p.active:
        return 0
    need = max((a.energy_cost for a in p.active.card.attacks), default=0)
    for c in p.hand:
        if not (is_pokemon(c) and getattr(c, "evolves_from", None) == p.active.card.id):
            continue
        need = max(need, max((a.energy_cost for a in c.attacks), default=0))
        break
    return need


def _try_attach_energy_auto(state: GameState) -> bool:
    """自動ターン用：エネルギーを 1 枚付与するなら True を返す。"""
    p = state.active_player_state()
    can_evolve_this_turn = state.turn_count > 1
    energy_hand_idx = next((i for i, c in enumerate(p.hand) if is_energy(c)), None)
    if energy_hand_idx is None or not p.active:
        return False

    if p.active.attached_energy == 0:
        attach_energy(state, energy_hand_idx)
        return True
    if can_evolve_this_turn and _should_attach_for_evolution(p):
        attach_energy(state, energy_hand_idx)
        return True

    energy_needed_active = _energy_needed_for_active(p)
    if p.active.attached_energy >= energy_needed_active and p.bench:
        bench_candidates = [
            (bi, b.attached_energy, b.card.attacks)
            for bi, b in enumerate(p.bench)
        ]
        max_need = 2
        need_more = [
            (bi, max_need - en, max((a.energy_cost for a in atks), default=0))
            for bi, en, atks in bench_candidates
        ]
        best_bench = max(need_more, key=lambda x: (x[1] > 0, x[2]), default=None)
        if best_bench and best_bench[1] > 0:
            attach_energy(state, energy_hand_idx, bench_index=best_bench[0])
            return True
    if p.active.attached_energy < energy_needed_active or not p.bench:
        attach_energy(state, energy_hand_idx)
        return True
    if p.bench:
        attach_energy(state, energy_hand_idx, bench_index=0)
        return True
    return False


def end_turn(state: GameState) -> None:
    """ターン終了：相手に交代、ターン数増加。"""
    state.log(f"{state.player_name(state.current_player)} のターン終了")
    state.current_player = state.opponent()
    state.turn_count += 1
    state.log("")


def run_turn_auto(state: GameState) -> bool:
    """
    現在のプレイヤーが「可能な行動を順番に実行」する。
    順序：0. 手札のポケモンをベンチに出す（5 体まで）、1. きずぐすり、2. エネルギー付与、3. 進化、4. 攻撃。
    何もできなければ False を返す。True = ターン内で何かした。
    """
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not p.active:
        return False
    
    acted = False

    while _put_one_pokemon_on_bench(p, state, state.current_player):
        acted = True

    opp_max_dmg = _opponent_max_damage(state)
    our_max_dmg = _our_max_damage(state)
    would_be_koed = p.active and opp_max_dmg > 0 and p.active.hp <= opp_max_dmg
    can_ko_opponent = opp.active and our_max_dmg >= opp.active.hp
    retreat_cost = getattr(p.active.card, "retreat_cost", 1) if p.active else 0
    if p.active and p.bench and would_be_koed and not can_ko_opponent and p.active.attached_energy >= retreat_cost + 100:
        best = max(
            range(len(p.bench)),
            key=lambda i: (p.bench[i].hp, p.bench[i].attached_energy),
            default=None,
        )
        if best is not None and retreat(state, best):
            acted = True
            p = state.active_player_state()

    if p.active and p.active.hp < p.active.max_hp:
        for i, c in enumerate(p.hand):
            if is_item(c) and getattr(c, "effect", None) == "heal":
                use_potion(state, i)
                acted = True
                break

    if _try_attach_energy_auto(state):
        acted = True
        p = state.active_player_state()

    is_first_turn_for_either = state.turn_count <= 1
    can_evolve = not is_first_turn_for_either
    if can_evolve:
        for hand_idx, c in enumerate(p.hand):
            if not is_pokemon(c) or not c.evolves_from:
                continue
            base_id = c.evolves_from
            if p.active and p.active.card.id == base_id:
                evolve_pokemon(state, hand_idx, bench_index=None)
                acted = True
                break
            for bench_idx, bench_poke in enumerate(p.bench):
                if bench_poke.card.id == base_id:
                    evolve_pokemon(state, hand_idx, bench_index=bench_idx)
                    acted = True
                    break
            if acted:
                break

    is_game_first_turn = state.turn_count == 0
    can_attack = not is_game_first_turn
    if can_attack:
        best_idx = _choose_best_attack_index(state, p, opp)
        if best_idx is not None:
            attack(state, best_idx)
            acted = True
    
    if not acted:
        state.log(f"{state.player_name(state.current_player)}: 実行できるアクションなし（パス）")
    
    return acted


def run_game_auto(state: GameState, max_turns: int | None = None) -> int | None:
    """
    自動でターンを進め、勝者が決まるまで実行。
    戻り値: 0 or 1 = 勝者、None = 最大ターンで未決着。
    """
    if max_turns is None:
        max_turns = MAX_TURNS
    if _check_game_end(state):
        if state.log_fn:
            state.log("========== ゲーム終了 ==========\n")
        return state.winner
    while state.turn_count < max_turns:
        start_turn(state)
        if state.winner is not None:
            if state.log_fn:
                state.log("========== ゲーム終了 ==========\n")
            return state.winner
        acted = run_turn_auto(state)
        if _check_game_end(state):
            if state.log_fn:
                state.log("========== ゲーム終了 ==========\n")
            return state.winner
        if not acted:
            end_turn(state)
        else:
            end_turn(state)
    if state.log_fn:
        state.log("最大ターンに達し未決着\n")
    return None
