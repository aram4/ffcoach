#!/usr/bin/env python3
"""
ffcoach — a weekly fantasy football assistant for ESPN leagues.

Reads every roster in your league, then produces:
  1. Matchup outlook   — your projection vs. your opponent's, rough win odds, their weak spots
  2. Start/sit         — the optimal lineup for this week vs. what you have set now
  3. Waiver targets    — free agents that improve your lineup, with who to drop
  4. Trade ideas       — 1-for-1 and 2-for-1 deals that improve BOTH starting lineups

Usage:
  python ffcoach.py                 # full markdown report (reads .env / env vars)
  python ffcoach.py --sms           # short text-message version (for OpenClaw)
  python ffcoach.py --narrate       # add an LLM-written brief on top (needs ANTHROPIC_API_KEY)
  python ffcoach.py --demo          # run against a fake league, no credentials needed
  python ffcoach.py --json out.json # also dump raw analysis for other tools
"""
from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings("ignore", message=".*OpenSSL.*")
import itertools
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field, asdict

# ----------------------------------------------------------------------------
# Tunables — adjust to taste
# ----------------------------------------------------------------------------
INJURY_WEEK_MULT = {  # how much of a player's projection to trust this week
    "OUT": 0.0, "INJURY_RESERVE": 0.0, "SUSPENSION": 0.0,
    "DOUBTFUL": 0.25, "QUESTIONABLE": 0.85,
}
INJURY_ROS_MULT = {"INJURY_RESERVE": 0.5, "OUT": 0.85, "SUSPENSION": 0.7}
ROS_PROJ_WEIGHT = 0.6        # ROS value = 0.6 * ESPN season projection avg + 0.4 * actual avg
TRADE_FAIRNESS = 0.25        # max relative gap in total value for a trade to look "fair"
TRADE_THEIR_MIN_GAIN = 0.3   # their starting lineup must improve too, or they won't accept
MATCHUP_SD = 24.0            # std-dev of the score *difference* used for rough win odds
SKIP_SLOTS = {"BE", "IR"}
TRADEABLE_POS = {"QB", "RB", "WR", "TE"}  # don't propose K / D/ST trades


# ----------------------------------------------------------------------------
# Data model (decoupled from espn_api so the logic is testable)
# ----------------------------------------------------------------------------
@dataclass
class P:
    name: str
    pos: str
    pro_team: str
    team_id: int | None          # None = free agent
    eligible: list[str]          # lineup slot labels this player can fill
    week_proj: float             # ESPN projection for this week
    ros: float                   # rest-of-season points/week estimate
    injury: str = "ACTIVE"
    on_bye: bool = False
    opp: str = ""
    opp_rank: int = 0            # opponent's rank vs this position (1 = toughest)
    owned: float = 0.0
    slot: str = "BE"             # current lineup slot
    kickoff: str = ""            # local kickoff time for this week's game, e.g. "Sun 12:00PM"

    @property
    def week_value(self) -> float:
        if self.on_bye:
            return 0.0
        return self.week_proj * INJURY_WEEK_MULT.get(self.injury, 1.0)

    @property
    def ros_value(self) -> float:
        return self.ros * INJURY_ROS_MULT.get(self.injury, 1.0)

    def tag(self) -> str:
        bits = []
        if self.on_bye:
            bits.append("BYE")
        if self.injury not in ("ACTIVE", "", "NORMAL", None):
            bits.append(self.injury.replace("INJURY_RESERVE", "IR").title())
        return f" ({', '.join(bits)})" if bits else ""


@dataclass
class Team:
    team_id: int
    name: str
    record: str
    roster: list[P]


@dataclass
class League:
    name: str
    week: int
    slots: dict[str, int]        # starting slots, e.g. {"QB":1,"RB":2,"RB/WR/TE":1,...}
    teams: dict[int, Team]
    matchups: dict[int, int]     # team_id -> opponent team_id for this week
    free_agents: list[P] = field(default_factory=list)
    espn: object = field(default=None, repr=False)   # raw espn_api League, for history lookups


# ----------------------------------------------------------------------------
# Lineup optimizer
# ----------------------------------------------------------------------------
def _slot_order(slots: dict[str, int]) -> list[str]:
    """Fill dedicated slots first, then flex slots from narrowest to widest."""
    labels = [s for s, n in slots.items() if n > 0 and s not in SKIP_SLOTS]
    def width(s: str) -> int:
        if s == "OP":
            return 10
        return s.count("/") + 1
    return sorted(labels, key=width)


def best_lineup(players: list[P], slots: dict[str, int], key) -> tuple[float, dict[str, list[P]]]:
    """Greedy fill: each slot takes the best still-unused eligible player.
    Filling narrow slots first makes greedy near-optimal for standard fantasy rosters."""
    used: set[int] = set()
    lineup: dict[str, list[P]] = {}
    total = 0.0
    ranked = sorted(players, key=key, reverse=True)
    for slot in _slot_order(slots):
        lineup[slot] = []
        for _ in range(slots[slot]):
            for p in ranked:
                if id(p) in used or slot not in p.eligible:
                    continue
                used.add(id(p))
                lineup[slot].append(p)
                total += key(p)
                break
    return total, lineup


def lineup_value(players, slots, key) -> float:
    return best_lineup(players, slots, key)[0]


def ros_key(p: P) -> float:
    return p.ros_value


def week_key(p: P) -> float:
    return p.week_value


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def analyze_matchup(lg: League, me: Team) -> dict:
    opp = lg.teams[lg.matchups[me.team_id]]
    my_best, my_lineup = best_lineup(me.roster, lg.slots, week_key)
    opp_best, opp_lineup = best_lineup(opp.roster, lg.slots, week_key)
    diff = my_best - opp_best
    win_prob = 0.5 * (1 + math.erf(diff / (MATCHUP_SD * math.sqrt(2))))

    # Opponent weak spots: their starters who are hurt or on bye
    opp_issues = [
        f"{p.name} {p.pos}{p.tag()}"
        for p in opp.roster
        if p.slot not in SKIP_SLOTS and (p.on_bye or INJURY_WEEK_MULT.get(p.injury, 1.0) < 1.0)
    ]
    if win_prob >= 0.65:
        strategy = "You're the favorite. Play it safe: start high-floor players, avoid boom/bust gambles."
    elif win_prob <= 0.35:
        strategy = "You're the underdog. Lean toward high-ceiling players and favorable matchups over safe floors."
    else:
        strategy = "Coin-flip week. Small edges matter: check Sunday-morning injury news before lock."
    return {
        "opponent": opp.name, "opp_record": opp.record,
        "my_proj": round(my_best, 1), "opp_proj": round(opp_best, 1),
        "win_prob": round(win_prob, 2), "strategy": strategy, "opp_issues": opp_issues,
        "_my_lineup": my_lineup,
    }


def analyze_start_sit(lg: League, me: Team, optimal: dict[str, list[P]]) -> dict:
    current_starters = [p for p in me.roster if p.slot not in SKIP_SLOTS]
    current_pts = sum(p.week_value for p in current_starters)
    optimal_players = [p for ps in optimal.values() for p in ps]
    optimal_pts = sum(p.week_value for p in optimal_players)
    opt_ids = {id(p) for p in optimal_players}
    cur_ids = {id(p) for p in current_starters}
    bench_these = [p for p in current_starters if id(p) not in opt_ids]
    start_these = [p for p in optimal_players if id(p) not in cur_ids]
    # Warnings on players in the optimal lineup who carry risk
    risky = [p for p in optimal_players if p.injury in ("QUESTIONABLE", "DOUBTFUL")]
    return {
        "current_pts": round(current_pts, 1), "optimal_pts": round(optimal_pts, 1),
        "gain": round(optimal_pts - current_pts, 1),
        "start": [(p.name, p.pos, round(p.week_value, 1)) for p in start_these],
        "bench": [(p.name, p.pos, round(p.week_value, 1), p.tag().strip(" ()")) for p in bench_these],
        "risky": [(p.name, p.injury.title()) for p in risky],
        "lineup": {s: [(p.name, round(p.week_value, 1), p.opp, p.opp_rank) for p in ps]
                   for s, ps in optimal.items()},
    }


def _drop_candidate(roster: list[P], slots, protect: P | None = None,
                    protect_this_week: bool = False) -> tuple[P, float]:
    """The player whose removal costs the least ROS lineup value (ties -> lowest ros).
    With protect_this_week, never drop someone needed in this week's lineup."""
    base = lineup_value(roster, slots, ros_key)
    base_wk = lineup_value(roster, slots, week_key) if protect_this_week else 0.0
    best = None
    for p in roster:
        if p is protect:
            continue
        rest = [q for q in roster if q is not p]
        loss = base - lineup_value(rest, slots, ros_key)
        wk_loss = base_wk - lineup_value(rest, slots, week_key) if protect_this_week else 0.0
        cand = (round(wk_loss, 2), loss, p.ros_value)
        if best is None or cand < best[0]:
            best = (cand, p)
    return best[1], best[0][1]


def analyze_waivers(lg: League, me: Team, top_n: int = 4) -> list[dict]:
    """Greedy, sequential plan: pick the best add/drop, apply it, then find the next best
    against the updated roster. So the moves stack and never drop the same player twice."""
    roster = list(me.roster)
    pool = [fa for fa in lg.free_agents if fa.ros_value > 0 or fa.week_value > 0]
    plan = []
    for _ in range(top_n):
        base_ros = lineup_value(roster, lg.slots, ros_key)
        base_wk = lineup_value(roster, lg.slots, week_key)
        best = None
        for fa in pool:
            cand = roster + [fa]
            drop, _ = _drop_candidate(cand, lg.slots, protect=fa)
            new = [q for q in cand if q is not drop]
            ros_gain = lineup_value(new, lg.slots, ros_key) - base_ros
            wk_gain = lineup_value(new, lg.slots, week_key) - base_wk
            if ros_gain <= 0.3 and wk_gain <= 1.0:
                continue
            score = ros_gain * 2 + wk_gain   # value the rest of the season over one week
            if best is None or score > best[0]:
                best = (score, fa, drop, new, ros_gain, wk_gain)
        if best is None:
            break
        _, fa, drop, roster, ros_gain, wk_gain = best
        pool.remove(fa)
        plan.append({
            "add": fa.name, "pos": fa.pos, "team": fa.pro_team, "owned": fa.owned,
            "drop": drop.name, "drop_pos": drop.pos,
            "ros_gain": round(ros_gain, 1), "week_gain": round(wk_gain, 1),
            "kind": "long-term add" if ros_gain >= 1.5 else "this-week streamer",
        })

    # Lineup holes (a starter projected for 0: bye, out) always get a fill suggestion
    _, lu = best_lineup(roster, lg.slots, week_key)
    for slot, ps in lu.items():
        for hole in [p for p in ps if p.week_value <= 0.1]:
            fills = [fa for fa in pool if slot in fa.eligible and fa.week_value > 0]
            if not fills:
                continue
            fa = max(fills, key=week_key)
            drop, _ = _drop_candidate(roster + [fa], lg.slots, protect=fa, protect_this_week=True)
            roster = [q for q in roster if q is not drop] + [fa]
            plan.append({
                "add": fa.name, "pos": fa.pos, "team": fa.pro_team, "owned": fa.owned,
                "drop": drop.name, "drop_pos": drop.pos, "ros_gain": 0.0,
                "week_gain": round(fa.week_value - hole.week_value, 1),
                "kind": f"fills empty {slot} slot ({hole.name}{hole.tag()})",
            })
            pool.remove(fa)
    return plan


def _trim_roster(roster: list[P], size: int, slots) -> list[P]:
    """If a team receives more players than it sends, assume it drops its least useful player."""
    roster = list(roster)
    while len(roster) > size:
        drop, _ = _drop_candidate(roster, slots)
        roster.remove(drop)
    return roster


def analyze_trades(lg: League, me: Team, top_n: int = 5, per_team: int = 2) -> list[dict]:
    my_base = lineup_value(me.roster, lg.slots, ros_key)
    mine = [p for p in me.roster if p.pos in TRADEABLE_POS]
    ideas = []
    for t in lg.teams.values():
        if t.team_id == me.team_id:
            continue
        their_base = lineup_value(t.roster, lg.slots, ros_key)
        theirs = [p for p in t.roster if p.pos in TRADEABLE_POS]
        # give 1 get 1, give 2 get 1 (consolidation), give 1 get 2 (depth)
        shapes = [(1, 1), (2, 1), (1, 2)]
        team_ideas = []
        for give_n, get_n in shapes:
            for give in itertools.combinations(mine, give_n):
                give_val = sum(p.ros_value for p in give)
                for get in itertools.combinations(theirs, get_n):
                    get_val = sum(p.ros_value for p in get)
                    hi = max(give_val, get_val)
                    if hi <= 0 or abs(give_val - get_val) / hi > TRADE_FAIRNESS:
                        continue
                    my_new = _trim_roster([p for p in me.roster if p not in give] + list(get),
                                          len(me.roster), lg.slots)
                    their_new = _trim_roster([p for p in t.roster if p not in get] + list(give),
                                             len(t.roster), lg.slots)
                    my_gain = lineup_value(my_new, lg.slots, ros_key) - my_base
                    their_gain = lineup_value(their_new, lg.slots, ros_key) - their_base
                    if my_gain < 0.5 or their_gain < TRADE_THEIR_MIN_GAIN:
                        continue
                    team_ideas.append({
                        "team": t.name, "record": t.record,
                        "give": [f"{p.name} ({p.pos})" for p in give],
                        "get": [f"{p.name} ({p.pos})" for p in get],
                        "my_gain": round(my_gain, 1), "their_gain": round(their_gain, 1),
                        "pitch": _pitch(give, get, t, lg),
                        # favor my gain, but reward deals they'd actually accept
                        "_score": my_gain + 0.5 * min(their_gain, my_gain),
                    })
        team_ideas.sort(key=lambda d: d["_score"], reverse=True)
        # don't repeat the same player (given or received) within one team's ideas
        seen: set[str] = set()
        for d in team_ideas:
            names = set(d["get"]) | set(d["give"])
            if names & seen:
                continue
            seen |= names
            ideas.append(d)
            if sum(1 for x in ideas if x["team"] == t.name) >= per_team:
                break
    ideas.sort(key=lambda d: d["_score"], reverse=True)
    return ideas[:top_n]


def _pitch(give, get, their_team: Team, lg: League) -> str:
    """One-line reason the other manager should care."""
    their_pos_counts: dict[str, int] = {}
    for p in their_team.roster:
        if p.injury not in ("INJURY_RESERVE", "OUT") and not p.on_bye:
            their_pos_counts[p.pos] = their_pos_counts.get(p.pos, 0) + 1
    need = [p.pos for p in give if their_pos_counts.get(p.pos, 0) <= lg.slots.get(p.pos, 1)]
    if need:
        return f"They're thin at {', '.join(sorted(set(need)))}; this fills a starting hole."
    if len(give) > len(get):
        return "They get depth for a crowded position."
    if len(get) > len(give):
        return "They consolidate two pieces into one stronger starter."
    return "Positional swap that helps both lineups."


# ----------------------------------------------------------------------------
# ESPN adapter
# ----------------------------------------------------------------------------
def load_espn(league_id: int, year: int, espn_s2: str | None, swid: str | None,
              my_team_id: int, fa_per_pos: int = 25) -> League:
    try:
        from espn_api.football import League as EspnLeague
    except ImportError:
        sys.exit("Missing dependency: pip install espn-api")

    lg = EspnLeague(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)
    week = lg.current_week
    slots = {k: v for k, v in lg.settings.position_slot_counts.items() if v and k not in SKIP_SLOTS}

    # Box-score players only carry this week's stats; season averages live on the team rosters.
    season = {p.playerId: p for t in lg.teams for p in t.roster}

    def ros_of(p) -> float:
        p = season.get(p.playerId, p)
        proj = getattr(p, "projected_avg_points", 0) or 0
        act = getattr(p, "avg_points", 0) or 0
        if proj and act:
            return ROS_PROJ_WEIGHT * proj + (1 - ROS_PROJ_WEIGHT) * act
        return proj or act

    def convert(bp, team_id) -> P:
        wk = getattr(bp, "projected_points", None)
        if wk is None:
            wk = bp.stats.get(week, {}).get("projected_points", 0)
        return P(
            name=bp.name, pos=bp.position, pro_team=bp.proTeam, team_id=team_id,
            eligible=list(bp.eligibleSlots), week_proj=float(wk or 0), ros=float(ros_of(bp)),
            injury=(bp.injuryStatus or "ACTIVE"), on_bye=bool(getattr(bp, "on_bye_week", False)),
            opp=str(getattr(bp, "pro_opponent", "")), opp_rank=int(getattr(bp, "pro_pos_rank", 0) or 0),
            owned=float(getattr(bp, "percent_owned", 0) or 0),
            slot=getattr(bp, "slot_position", "BE"),
            kickoff=gd.strftime("%a %-I:%M%p") if (gd := getattr(bp, "game_date", None)) else "",
        )

    teams: dict[int, Team] = {}
    matchups: dict[int, int] = {}
    for bs in lg.box_scores(week):
        for side, other in (("home", "away"), ("away", "home")):
            t = getattr(bs, f"{side}_team")
            o = getattr(bs, f"{other}_team")
            if not t or isinstance(t, int):
                continue  # bye week in odd-team leagues
            teams[t.team_id] = Team(
                team_id=t.team_id, name=t.team_name, record=f"{t.wins}-{t.losses}",
                roster=[convert(p, t.team_id) for p in getattr(bs, f"{side}_lineup")],
            )
            if o and not isinstance(o, int):
                matchups[t.team_id] = o.team_id

    fas, seen = [], set()
    for pos in ("QB", "RB", "WR", "TE", "K", "D/ST"):
        for bp in lg.free_agents(week=week, size=fa_per_pos, position=pos):
            if bp.playerId in seen:
                continue
            seen.add(bp.playerId)
            fas.append(convert(bp, None))

    if my_team_id not in teams:
        names = ", ".join(f"{t.team_id}={t.name}" for t in teams.values())
        sys.exit(f"Team id {my_team_id} not found. Teams in league: {names}")
    if my_team_id not in matchups:
        sys.exit(f"No matchup found for your team in week {week} (bye week?).")

    return League(name=lg.settings.name, week=week, slots=slots,
                  teams=teams, matchups=matchups, free_agents=fas, espn=lg)


HISTORY_TYPES = ["FREEAGENT", "WAIVER", "WAIVER_ERROR", "ROSTER", "TRADE_PROPOSAL",
                 "TRADE_ACCEPT", "TRADE_DECLINE", "TRADE_VETO", "TRADE_UPHOLD"]


def load_history(lg: League) -> dict:
    """Every manager's moves this season: waiver bids (won and lost), free-agent pickups,
    completed trades, trade offers and responses, lineup activity and early draft picks."""
    from datetime import datetime
    esp = lg.espn
    team_name = {t.team_id: t.name for t in lg.teams.values()}
    pos = {p.name: p.pos for t in lg.teams.values() for p in t.roster}
    pos.update({p.name: p.pos for p in lg.free_agents})

    def player(pid):
        n = esp.player_map.get(pid, str(pid))
        return f"{n} ({pos[n]})" if n in pos else n

    def day(ms):
        return datetime.fromtimestamp(ms / 1000).strftime("%b %d") if ms else None

    raw = {}
    # ESPN files transactions by scoring period (week); 0 is preseason.
    for wk in range(esp.current_week + 1):
        data = esp.espn_request.league_get(
            params={"view": "mTransactions2", "scoringPeriodId": wk},
            headers={"x-fantasy-filter": json.dumps({"transactions": {"filterType": {"value": HISTORY_TYPES}}})})
        for t in data.get("transactions", []):
            raw[t["id"]] = t

    budget = getattr(esp.settings, "acquisition_budget", 0) or 0
    mgrs = {}
    for et in esp.teams:
        mgrs[et.team_id] = {
            "team": team_name.get(et.team_id, et.team_name),
            "manager": ", ".join(f"{o.get('firstName', '').strip()} {o.get('lastName', '').strip()}".strip() for o in et.owners),
            "record": f"{et.wins}-{et.losses}",
            "playoff_seed": et.standing, "waiver_priority": et.waiver_rank,
            "faab_left": budget - et.acquisition_budget_spent if budget else None,
            "lineup_changes_by_week": {}, "waiver_bids": [], "free_agent_moves": [],
            "trades_completed": [], "trade_offers_made": [], "trade_offers_received": [],
            "trade_responses": {"accepted": 0, "declined": 0},
            "early_draft_picks": [],
        }
    contests: dict[str, list] = {}

    for t in sorted(raw.values(), key=lambda t: t.get("proposedDate") or t.get("processDate") or 0):
        m = mgrs.get(t["teamId"])
        if not m:
            continue
        items = t.get("items") or []
        adds = [player(i["playerId"]) for i in items if i["type"] == "ADD"]
        drops = [player(i["playerId"]) for i in items if i["type"] == "DROP"]
        wk, status, typ = t.get("scoringPeriodId"), t.get("status"), t["type"]
        if typ == "WAIVER":
            won = status == "EXECUTED"
            bid = {"week": wk, "add": adds[0] if adds else None, "drop": drops or None,
                   "bid": t.get("bidAmount"), "result": "won" if won else status.replace("FAILED_", "lost: ").lower()}
            m["waiver_bids"].append(bid)
            if adds:
                contests.setdefault(f"wk{wk} {adds[0]}", []).append(
                    {"team": m["team"], "bid": t.get("bidAmount"), "won": won})
        elif typ == "FREEAGENT" and status == "EXECUTED":
            m["free_agent_moves"].append({"week": wk, "add": adds, "drop": drops})
        elif typ == "ROSTER" and any(i["type"] == "LINEUP" for i in items):
            m["lineup_changes_by_week"][wk] = m["lineup_changes_by_week"].get(wk, 0) + 1
        elif typ == "TRADE_PROPOSAL" and items:
            sides: dict[int, list] = {}
            for i in items:
                if i["type"] == "ACQUISITION_BUDGET_TRADE":
                    what = f"${i.get('bidAmount', '?')} FAAB"
                else:
                    what = player(i["playerId"]) if i["playerId"] else "draft pick"
                sides.setdefault(i["fromTeamId"], []).append(what)
            other = next((tid for tid in sides if tid != t["teamId"]), None)
            # ESPN only shows some offers, and their outcome isn't always linked, so `status` is as-is.
            offer = {"week": wk, "date": day(t.get("proposedDate")), "status": status.lower() if status else None,
                     "from": m["team"], "to": team_name.get(other),
                     "offered": sides.get(t["teamId"], []), "asked_for": sides.get(other, [])}
            m["trade_offers_made"].append(offer)
            if other in mgrs:
                mgrs[other]["trade_offers_received"].append(offer)
        elif typ == "TRADE_ACCEPT":
            m["trade_responses"]["accepted"] += 1
        elif typ == "TRADE_DECLINE":
            m["trade_responses"]["declined"] += 1

    # Completed trades are clearest in the activity feed, which names both sides.
    offset = 0
    while True:
        batch = esp.recent_activity(size=100, offset=offset, msg_type="TRADED")
        for a in batch:
            got: dict[str, list] = {}
            for tm, action, p, _ in a.actions:
                if action == "TRADE_RECEIVED" and tm:
                    got.setdefault(tm.team_name, []).append(f"{p.name} ({p.position})" if hasattr(p, "name") else str(p))
            for et in esp.teams:
                if et.team_name in got:
                    mgrs[et.team_id]["trades_completed"].append({"date": day(a.date), "received": got})
        if len(batch) < 100:
            break
        offset += 100

    for pick in esp.draft:
        m = mgrs.get(pick.team.team_id) if pick.team else None
        if m and pick.round_num <= 6:
            m["early_draft_picks"].append(
                {"round": pick.round_num, "player": f"{pick.playerName} ({pos.get(pick.playerName, '?')})",
                 **({"auction_price": pick.bid_amount} if pick.bid_amount else {})})

    return {
        "faab_budget": budget or None,
        "managers": list(mgrs.values()),
        "contested_waiver_claims": {k: v for k, v in contests.items() if len(v) > 1},
    }


# ----------------------------------------------------------------------------
# Demo league (for testing without credentials)
# ----------------------------------------------------------------------------
def load_demo(seed: int = 7) -> tuple[League, int]:
    rng = random.Random(seed)
    slots = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR/TE": 1, "D/ST": 1, "K": 1}
    elig = {
        "QB": ["QB", "OP", "BE", "IR"], "RB": ["RB", "RB/WR", "RB/WR/TE", "OP", "BE", "IR"],
        "WR": ["WR", "RB/WR", "WR/TE", "RB/WR/TE", "OP", "BE", "IR"],
        "TE": ["TE", "WR/TE", "RB/WR/TE", "OP", "BE", "IR"],
        "K": ["K", "BE", "IR"], "D/ST": ["D/ST", "BE", "IR"],
    }
    base = {"QB": 19, "RB": 15, "WR": 14, "TE": 10, "K": 8, "D/ST": 7}
    spread = {"QB": 6, "RB": 8, "WR": 8, "TE": 5, "K": 2, "D/ST": 3}
    pros = ["KC", "BUF", "PHI", "DET", "SF", "BAL", "CIN", "MIA", "DAL", "GB", "HOU", "LAR",
            "MIN", "SEA", "ATL", "CHI", "PIT", "NYJ", "LAC", "TB"]
    on_bye = set(rng.sample(pros, 4))
    first = ["Jalen", "Marcus", "Tyler", "Devon", "Chris", "Andre", "Jordan", "Malik", "Kyle",
             "Trey", "Isaiah", "Brandon", "Cole", "Darius", "Evan", "Nico"]
    last = ["Brooks", "Hayes", "Coleman", "Reed", "Fields", "Mason", "Price", "Carter", "Bell",
            "Grant", "Hill", "Owens", "Shaw", "Ward", "Lane", "Ford", "Webb", "Stone"]
    injuries = ["ACTIVE"] * 14 + ["QUESTIONABLE", "QUESTIONABLE", "DOUBTFUL", "OUT", "INJURY_RESERVE"]
    counter = itertools.count()

    def make(pos, tier, team_id):
        n = next(counter)
        pro = rng.choice(pros)
        ros = max(1.0, base[pos] + spread[pos] * tier + rng.gauss(0, 1.5))
        wk = max(0.0, ros + rng.gauss(0, ros * 0.2))
        name = f"{first[n % len(first)]} {last[(n * 7) % len(last)]}" if pos != "D/ST" else f"{pro} D/ST"
        return P(name=name, pos=pos, pro_team=pro, team_id=team_id, eligible=elig[pos],
                 week_proj=round(wk, 1), ros=round(ros, 1), injury=rng.choice(injuries),
                 on_bye=pro in on_bye, opp=rng.choice(pros), opp_rank=rng.randint(1, 32),
                 owned=round(rng.uniform(1, 99), 1))

    team_names = ["Tuesday Night Waivers", "Mahomes Alone", "CMC Hammer", "Bijan Mustard",
                  "The Kelce Kids", "Lamb Chops", "Puka Shells", "Bucky Irving Ave",
                  "Saquon Sense", "Hurts So Good"]
    shape = {"QB": 2, "RB": 5, "WR": 5, "TE": 2, "K": 1, "D/ST": 1}
    teams: dict[int, Team] = {}
    for tid, nm in enumerate(team_names, start=1):
        roster = []
        for pos, n in shape.items():
            for i in range(n):
                tier = rng.uniform(-0.6, 1.0) - i * 0.35
                roster.append(make(pos, tier, tid))
        # set a slightly-wrong "current" lineup the way a busy manager would
        _, lu = best_lineup(roster, slots, ros_key)
        for s, ps in lu.items():
            for p in ps:
                p.slot = s
        w, l = rng.randint(1, 4), 0
        teams[tid] = Team(tid, nm, f"{w}-{4 - w}", roster)
    fas = []
    for pos, n in {"QB": 6, "RB": 10, "WR": 10, "TE": 6, "K": 5, "D/ST": 5}.items():
        for _ in range(n):
            p = make(pos, rng.uniform(-1.3, 0.3), None)
            p.owned = round(rng.uniform(0.5, 40), 1)
            fas.append(p)
    ids = list(teams)
    matchups = {}
    for a, b in zip(ids[::2], ids[1::2]):
        matchups[a], matchups[b] = b, a
    lg = League(name="Demo League", week=5, slots=slots, teams=teams,
                matchups=matchups, free_agents=fas)
    return lg, 1


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def build_report(lg: League, my_id: int) -> dict:
    me = lg.teams[my_id]
    m = analyze_matchup(lg, me)
    ss = analyze_start_sit(lg, me, m.pop("_my_lineup"))
    return {
        "league": lg.name, "week": lg.week, "team": me.name, "record": me.record,
        "matchup": m, "start_sit": ss,
        "waivers": analyze_waivers(lg, me),
        "trades": analyze_trades(lg, me),
    }


def to_markdown(r: dict) -> str:
    m, ss = r["matchup"], r["start_sit"]
    L = [f"# Week {r['week']} brief: {r['team']} ({r['record']})", ""]
    L += ["## Matchup", f"vs **{m['opponent']}** ({m['opp_record']}). "
          f"Optimal projections: **{m['my_proj']} – {m['opp_proj']}**, "
          f"rough win odds **{int(m['win_prob'] * 100)}%**.", "", m["strategy"]]
    if m["opp_issues"]:
        L += ["", "Their lineup problems: " + "; ".join(m["opp_issues"])]

    L += ["", "## Start / sit"]
    if ss["gain"] > 0.05:
        L.append(f"Your current lineup leaves **{ss['gain']} pts** on the table "
                 f"({ss['current_pts']} → {ss['optimal_pts']}).")
        for name, pos, pts in ss["start"]:
            L.append(f"- **Start** {name} ({pos}, {pts} proj)")
        for name, pos, pts, tag in ss["bench"]:
            L.append(f"- **Bench** {name} ({pos}, {pts} proj{', ' + tag if tag else ''})")
    else:
        L.append(f"Your lineup is already optimal ({ss['optimal_pts']} proj).")
    if ss["risky"]:
        L.append("- Watch before lock: " + ", ".join(f"{n} ({s})" for n, s in ss["risky"]))
    L += ["", "| Slot | Player | Proj | Opp (rank vs pos) |", "|---|---|---|---|"]
    for slot, ps in ss["lineup"].items():
        for name, pts, opp, rank in ps:
            L.append(f"| {slot} | {name} | {pts} | {opp}{f' ({rank})' if rank else ''} |")

    L += ["", "## Waiver targets"]
    if r["waivers"]:
        L.append("_In priority order. Each move assumes the ones above it went through._")
        for w in r["waivers"]:
            L.append(f"- **Add {w['add']}** ({w['pos']}, {w['team']}, {w['owned']}% owned), "
                     f"drop {w['drop']} ({w['drop_pos']}): "
                     f"+{w['ros_gain']} pts/wk ROS, +{w['week_gain']} this week. *{w['kind']}*")
    else:
        L.append("Nothing on the wire beats your roster right now. Hold your priority.")

    L += ["", "## Trade ideas", "_Gains are rest-of-season starting-lineup pts/week for each side._"]
    if r["trades"]:
        for t in r["trades"]:
            L.append(f"- **{t['team']}** ({t['record']}): give {' + '.join(t['give'])} "
                     f"for {' + '.join(t['get'])}. You +{t['my_gain']}, them +{t['their_gain']}. "
                     f"{t['pitch']}")
    else:
        L.append("No win-win deals found. Your roster fits your lineup well, or the values don't line up.")
    return "\n".join(L)


def to_sms(r: dict) -> str:
    m, ss = r["matchup"], r["start_sit"]
    L = [f"Wk{r['week']} vs {m['opponent']}: {m['my_proj']}-{m['opp_proj']} ({int(m['win_prob'] * 100)}%)"]
    if ss["gain"] > 0.05:
        L.append("Lineup: start " + ", ".join(n for n, _, _ in ss["start"]) +
                 "; bench " + ", ".join(n for n, *_ in ss["bench"]) + f" (+{ss['gain']})")
    else:
        L.append("Lineup: already optimal")
    if ss["risky"]:
        L.append("Check: " + ", ".join(f"{n} {s[0]}" for n, s in ss["risky"]))
    for w in r["waivers"][:2]:
        L.append(f"Add {w['add']}/drop {w['drop']} (+{w['ros_gain']} ROS)")
    if r["trades"]:
        t = r["trades"][0]
        L.append(f"Trade: {'+'.join(g.split(' (')[0] for g in t['give'])} -> "
                 f"{'+'.join(g.split(' (')[0] for g in t['get'])} w/ {t['team']} (+{t['my_gain']})")
    return "\n".join(L)


def narrate(report: dict, model: str) -> str:
    try:
        import anthropic
    except ImportError:
        return "_(narration skipped: pip install anthropic)_"
    client = anthropic.Anthropic()
    clean = json.loads(json.dumps(report, default=str))
    prompt = (
        "You are a sharp, concise fantasy football advisor. Using ONLY the analysis JSON below, "
        "write a 150-word weekly brief: the one or two moves that matter most, why, and anything to "
        "re-check before kickoff. Don't invent stats or news that aren't in the data.\n\n"
        + json.dumps(clean, indent=1)
    )
    msg = client.messages.create(model=model, max_tokens=600,
                                 messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def _load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    ap = argparse.ArgumentParser(description="Weekly ESPN fantasy football assistant")
    ap.add_argument("--demo", action="store_true", help="use a fake league (no credentials)")
    ap.add_argument("--sms", action="store_true", help="short text-message output")
    ap.add_argument("--narrate", action="store_true", help="add an LLM-written brief")
    ap.add_argument("--model", default=os.environ.get("FFCOACH_MODEL", "claude-sonnet-5-5"))
    ap.add_argument("--json", metavar="PATH", help="also write raw analysis JSON here")
    args = ap.parse_args()

    if args.demo:
        lg, my_id = load_demo()
    else:
        _load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
        need = ["ESPN_LEAGUE_ID", "ESPN_TEAM_ID"]
        missing = [k for k in need if not os.environ.get(k)]
        if missing:
            sys.exit(f"Set {', '.join(missing)} in .env (see README). Or try --demo.")
        lg = load_espn(
            league_id=int(os.environ["ESPN_LEAGUE_ID"]),
            year=int(os.environ.get("ESPN_YEAR", "2026")),
            espn_s2=os.environ.get("ESPN_S2"), swid=os.environ.get("ESPN_SWID"),
            my_team_id=int(os.environ["ESPN_TEAM_ID"]),
        )
        my_id = int(os.environ["ESPN_TEAM_ID"])

    report = build_report(lg, my_id)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2, default=str)

    out = to_sms(report) if args.sms else to_markdown(report)
    if args.narrate:
        out = narrate(report, args.model) + "\n\n---\n\n" + out
    print(out)


if __name__ == "__main__":
    main()
