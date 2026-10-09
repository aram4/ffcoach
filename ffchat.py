#!/usr/bin/env python3
"""
ffchat: ask questions about your ESPN fantasy league in plain English.

Claude reads your live league through a few tools and does the reasoning itself.
The only math done in code is trade scoring, where exact lineup arithmetic matters.

  python ffchat.py                        # interactive chat
  python ffchat.py "start Hill or Reed?"  # one question
  python ffchat.py --session s.json "..." # one question, remembering earlier ones
  python ffchat.py --demo                 # fake league
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message=".*OpenSSL.*")

import ffcoach as fc

MAX_TOOL_ROUNDS = 25
SESSION_TTL_MIN = 120

# Lookups go to Sonnet, strategy to Opus; Haiku picks which. --model forces one.
ROUTES = {
    "simple": ("claude-sonnet-5-5", os.environ.get("FFCOACH_SIMPLE_EFFORT", "medium")),
    "strategy": ("claude-opus-5-5", os.environ.get("FFCOACH_EFFORT", "high")),
}
ROUTER_MODEL = "claude-haiku-4-5"
ROUTER_PROMPT = """Classify a fantasy football question. Reply with one word.
strategy: needs judgment, e.g. trades, pickups/drops, start/sit, lineup advice, "what should I do".
simple: a factual lookup, e.g. a projection, injury, owner, roster, record, kickoff time.
If unsure, say strategy."""


class LeagueData:
    def __init__(self, demo: bool, team_id: int | None = None):
        self.demo = demo
        self.team_id = team_id
        self.load()

    def load(self):
        if self.demo:
            self.lg, self.my_id = fc.load_demo()
        else:
            need = ("ESPN_LEAGUE_ID",) if self.team_id else ("ESPN_LEAGUE_ID", "ESPN_TEAM_ID")
            missing = [k for k in need if not os.environ.get(k)]
            if missing:
                sys.exit(f"Set {', '.join(missing)} in .env (see README). Or try --demo.")
            self.my_id = self.team_id or int(os.environ["ESPN_TEAM_ID"])
            self.lg = fc.load_espn(
                league_id=int(os.environ["ESPN_LEAGUE_ID"]),
                year=int(os.environ.get("ESPN_YEAR", "2026")),
                espn_s2=os.environ.get("ESPN_S2"), swid=os.environ.get("ESPN_SWID"),
                my_team_id=self.my_id,
            )
        self.loaded_at = time.strftime("%a %H:%M")
        self.history = None

    @property
    def me(self) -> fc.Team:
        return self.lg.teams[self.my_id]

    def team(self, q: str) -> fc.Team:
        ql = q.lower()
        if ql in ("me", "my team", "mine"):
            return self.me
        if ql in ("opponent", "opp", "my opponent"):
            return self.lg.teams[self.lg.matchups[self.my_id]]
        names = {t.name.lower(): t for t in self.lg.teams.values()}
        match = next((t for n, t in names.items() if ql in n), None)
        if match:
            return match
        close = difflib.get_close_matches(ql, names, n=1, cutoff=0.4)
        if close:
            return names[close[0]]
        raise LookupError(f"No team matching '{q}'. Teams: {', '.join(t.name for t in self.lg.teams.values())}")

    def player(self, q: str) -> fc.P:
        players = [p for t in self.lg.teams.values() for p in t.roster] + self.lg.free_agents
        ql = q.lower().strip()
        exact = [p for p in players if p.name.lower() == ql]
        if exact:
            return exact[0]
        partial = [p for p in players if ql in p.name.lower()]
        if len(partial) == 1:
            return partial[0]
        if partial:
            opts = ", ".join(f"{p.name} ({p.pos}, {self.owner(p)})" for p in partial[:6])
            raise LookupError(f"'{q}' is ambiguous: {opts}. Use the full name.")
        close = difflib.get_close_matches(ql, [p.name.lower() for p in players], n=3, cutoff=0.6)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise LookupError(f"No player '{q}' on any roster or among the top free agents.{hint}")

    def owner(self, p: fc.P) -> str:
        if p.team_id is None:
            return "free agent"
        return "you" if p.team_id == self.my_id else self.lg.teams[p.team_id].name


def pdict(p: fc.P, d: LeagueData) -> dict:
    out = {
        "name": p.name, "pos": p.pos, "nfl_team": p.pro_team, "owner": d.owner(p),
        "slot": p.slot if p.team_id is not None else None,
        "week_proj": round(p.week_proj, 1), "ros_pts_per_week": round(p.ros_value, 1),
        "injury": p.injury if p.injury not in ("ACTIVE", "NORMAL") else None,
        "bye": p.on_bye or None, "opponent": p.opp if p.opp not in ("", "None") else None,
        "opp_rank_vs_pos": p.opp_rank or None, "kickoff": p.kickoff or None,
        "pct_owned": p.owned,
    }
    return {k: v for k, v in out.items() if v is not None}


TOOLS = [
    {"name": "league_overview",
     "description": "League name, week, starting lineup slots, every team's record, and this week's matchups.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_rosters",
     "description": "Rosters with each player's lineup slot, this-week projection, rest-of-season points/week, "
                    "injury, bye, opponent and kickoff. teams: names, 'me', 'opponent', or ['all'] for the whole league.",
     "input_schema": {"type": "object", "properties": {"teams": {"type": "array", "items": {"type": "string"}}},
                      "required": ["teams"]}},
    {"name": "find_players",
     "description": "Look up players by name anywhere in the league (rosters or free agents).",
     "input_schema": {"type": "object", "properties": {"names": {"type": "array", "items": {"type": "string"}}},
                      "required": ["names"]}},
    {"name": "free_agents",
     "description": "Best available free agents, optionally by position (QB, RB, WR, TE, K, D/ST), "
                    "sorted by this week ('week') or rest of season ('ros').",
     "input_schema": {"type": "object", "properties": {
         "position": {"type": "string"}, "sort": {"type": "string", "enum": ["week", "ros"]},
         "limit": {"type": "integer"}}}},
    {"name": "evaluate_trades",
     "description": "Score trades between the user and other teams: how each side's best starting lineup changes "
                    "in rest-of-season points/week and this-week points. Pass every idea you want checked in one call.",
     "input_schema": {"type": "object", "properties": {"trades": {"type": "array", "items": {
         "type": "object", "properties": {
             "give": {"type": "array", "items": {"type": "string"}},
             "get": {"type": "array", "items": {"type": "string"}}}, "required": ["give", "get"]}}},
         "required": ["trades"]}},
    {"name": "manager_history",
     "description": "How each manager behaves: every waiver bid (won and lost, with amounts), free-agent moves, "
                    "completed trades, trade offers made and received, accept/decline counts, lineup changes per "
                    "week, FAAB left, waiver priority and early draft picks. Also lists contested waiver claims with "
                    "every bid. teams: names or ['all'].",
     "input_schema": {"type": "object", "properties": {"teams": {"type": "array", "items": {"type": "string"}}},
                      "required": ["teams"]}},
    {"name": "refresh_data",
     "description": "Re-download league data from ESPN, e.g. after a trade or waiver claim.",
     "input_schema": {"type": "object", "properties": {}}},
    # Runs on Anthropic's servers; results come back inside the response, not through run_tool.
    {"type": "web_search_20260209", "name": "web_search", "max_uses": 10},
]


def run_tool(d: LeagueData, name: str, args: dict):
    lg = d.lg
    if name == "league_overview":
        return {
            "league": lg.name, "week": lg.week, "data_loaded": d.loaded_at,
            "starting_slots": lg.slots, "your_team": d.me.name,
            "teams": [{"name": t.name, "record": t.record,
                       "opponent": lg.teams[lg.matchups[t.team_id]].name if t.team_id in lg.matchups else None}
                      for t in lg.teams.values()],
        }
    if name == "get_rosters":
        teams = lg.teams.values() if args["teams"] == ["all"] else [d.team(q) for q in args["teams"]]
        return [{"team": t.name, "record": t.record, "roster": [pdict(p, d) for p in t.roster]} for t in teams]
    if name == "find_players":
        out = []
        for n in args["names"]:
            try:
                out.append(pdict(d.player(n), d))
            except LookupError as e:
                out.append({"query": n, "error": str(e)})
        return out
    if name == "free_agents":
        pos = (args.get("position") or "").upper().replace("DST", "D/ST")
        key = fc.ros_key if args.get("sort") == "ros" else fc.week_key
        pool = [p for p in lg.free_agents if not pos or p.pos == pos]
        return [pdict(p, d) for p in sorted(pool, key=key, reverse=True)[: args.get("limit", 10)]]
    if name == "evaluate_trades":
        out = []
        for t in args["trades"]:
            try:
                out.append({"give": t["give"], "get": t["get"], **evaluate_trade(d, t["give"], t["get"])})
            except LookupError as e:
                out.append({"give": t["give"], "get": t["get"], "error": str(e)})
        return out
    if name == "manager_history":
        if d.demo:
            return {"note": "No transaction history in demo mode."}
        if d.history is None:
            d.history = fc.load_history(lg)
        h = d.history
        if args["teams"] == ["all"]:
            return h
        wanted = {d.team(q).name for q in args["teams"]}
        return {**h, "managers": [m for m in h["managers"] if m["team"] in wanted]}
    if name == "refresh_data":
        d.load()
        return {"ok": True, "week": lg.week, "loaded": d.loaded_at}
    raise LookupError(f"Unknown tool {name}")


def evaluate_trade(d: LeagueData, give_names: list[str], get_names: list[str]) -> dict:
    lg, me = d.lg, d.me
    give = [d.player(n) for n in give_names]
    get = [d.player(n) for n in get_names]
    if any(p.team_id != d.my_id for p in give):
        raise LookupError("Every player in 'give' must be on the user's roster.")
    owners = {p.team_id for p in get}
    if len(owners) != 1 or None in owners or d.my_id in owners:
        raise LookupError("Every player in 'get' must be on the same other team.")
    them = lg.teams[owners.pop()]
    # A side that receives more players than it sends drops its least useful one.
    my_new = fc._trim_roster([p for p in me.roster if p not in give] + get, len(me.roster), lg.slots)
    th_new = fc._trim_roster([p for p in them.roster if p not in get] + give, len(them.roster), lg.slots)
    res = {"with_team": them.name}
    for label, key in (("ros", fc.ros_key), ("this_week", fc.week_key)):
        res[f"your_change_{label}"] = round(fc.lineup_value(my_new, lg.slots, key) - fc.lineup_value(me.roster, lg.slots, key), 1)
        res[f"their_change_{label}"] = round(fc.lineup_value(th_new, lg.slots, key) - fc.lineup_value(them.roster, lg.slots, key), 1)
    return res


def system_prompt(d: LeagueData, sms: bool, model: str) -> str:
    style = ("Reply in under 320 characters, plain text, no markdown: it's going out as a text message."
             if sms else "Lead with the answer, then the 2-5 points that drove it. Light markdown is fine.")
    return f"""You are ffchat, a fantasy football advisor for the manager of '{d.me.name}' in an ESPN league (week {d.lg.week}). Today is {time.strftime("%A, %B %d, %Y")}.

This answer comes from Anthropic's Claude model `{model}`. ffchat sends quick lookups to Claude Sonnet 5.5 (`claude-sonnet-5-5`) and strategy questions to Claude Opus 5.5 (`claude-opus-5-5`), so earlier answers may have come from the other model. Never "correct" an earlier answer about which model gave it.

How to work:
- Ground every fact (projections, injuries, byes, owners, slots, kickoffs) in tool results. Reuse earlier results in this conversation when they cover the question; otherwise call a tool first.
- Think like a strong manager. Weigh rest-of-season value against this week, positional scarcity, byes, injury risk, correlations with the user's own starters, and their record.
- Trades: read the other rosters to find teams with surplus where the user is thin and vice versa, then score all your candidate ideas in a single evaluate_trades call. A good offer helps the user and doesn't obviously hurt the other side, or they won't accept.
- Moves: compare free agents at every position against the user's weakest starters and bench.
- Model the other managers with manager_history. Before recommending a trade, judge how likely that specific manager is to accept: how often they trade, what they've given up and gone after, offers they've made (which show what they want), how they've responded to offers, their record and playoff position, and whether the deal fills a real hole for them. Rank trade ideas by value to the user and by how likely that manager is to accept, and say why.
- For waiver targets, predict who else will go after each player: managers with a roster need at that position, higher waiver priority or more FAAB left, and a history of bidding on similar players. Suggest FAAB bids based on what this league has actually paid in contested claims.
- Activity matters: a manager who rarely changes their lineup or makes moves is less likely to answer a trade offer at all.
- Lineups: build the best lineup yourself from projections and the starting slots. Use kickoff times for late-swap advice.
- ESPN data has injury tags but no news. Use web_search for the latest injury, practice, depth-chart and roster news on players that matter to the answer, especially anyone you're recommending to trade for, trade away, start or pick up. Prefer reports from the last few days, say how recent each one is, and let the news override stale ESPN projections (e.g. a player ruled out or placed on IR). Name the source briefly.
- If an earlier answer was wrong, correct it plainly. Never invent a mistake or a correction.
- week_proj is ESPN's projection. ros_pts_per_week blends ESPN's season projection with the actual average so far. Give one clear recommendation and say when a call is close.

{style}"""


def route(client, question: str, history: list, forced: str | None, verbose: bool) -> tuple[str, str]:
    if forced:
        return forced, ROUTES["strategy" if "opus" in forced else "simple"][1]
    prev = [m["content"] for m in history if m["role"] == "user" and isinstance(m["content"], str)]
    text = (f"Previous question: {prev[-1]}\n" if prev else "") + f"Question: {question}"
    kind = "strategy"
    try:
        r = client.messages.create(model=ROUTER_MODEL, max_tokens=5, system=ROUTER_PROMPT,
                                   messages=[{"role": "user", "content": text}])
        if "simple" in "".join(b.text for b in r.content if b.type == "text").lower():
            kind = "simple"
    except Exception:
        pass
    if verbose:
        print(f"  [{kind} → {ROUTES[kind][0]}]", file=sys.stderr)
    return ROUTES[kind]


def ask(client, model: str, effort: str, d: LeagueData, history: list, question: str,
        sms: bool, verbose: bool, log=None) -> str:
    def note(msg):
        if verbose:
            print(f"  · {msg}", file=sys.stderr)
        if log:
            log(msg)

    # history keeps every block (thinking, tool calls, results) and is only ever appended to:
    # Opus/Sonnet thinking blocks stay valid only if earlier turns are unchanged.
    history.append({"role": "user", "content": question})
    for _ in range(MAX_TOOL_ROUNDS):
        with client.messages.stream(
            model=model, max_tokens=32000, system=system_prompt(d, sms, model),
            tools=TOOLS, messages=history, output_config={"effort": effort},
            cache_control={"type": "ephemeral"},
        ) as stream:
            resp = stream.get_final_message()
        history.append({"role": "assistant",
                        "content": [b.model_dump(mode="json", exclude_none=True) for b in resp.content]})
        if resp.stop_reason == "pause_turn":   # long server-side search; resend to let it finish
            continue
        if resp.stop_reason == "refusal":
            return "Claude declined to answer that one. Try rephrasing."
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text").strip()
        results = []
        for b in resp.content:
            if b.type == "server_tool_use":
                note(f"{b.name}({json.dumps(b.input)})")
            if b.type != "tool_use":
                continue
            note(f"{b.name}({json.dumps(b.input)})")
            try:
                out = json.dumps(run_tool(d, b.name, b.input or {}), default=str)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            except Exception as e:
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": str(e), "is_error": True})
        history.append({"role": "user", "content": results})
    return "Sorry, that took too many steps. Try a narrower question."


def load_session(path):
    if not path or not os.path.exists(path):
        return []
    s = json.load(open(path))
    if time.time() - s.get("ts", 0) > SESSION_TTL_MIN * 60:
        return []
    return s.get("history", [])


def save_session(path, history):
    if path:
        json.dump({"ts": time.time(), "history": history}, open(path, "w"))


def main():
    ap = argparse.ArgumentParser(description="Ask questions about your ESPN fantasy league")
    ap.add_argument("question", nargs="*", help="one question; omit for interactive chat")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--sms", action="store_true", help="short plain-text answers")
    ap.add_argument("--session", metavar="PATH", help="file that keeps context between runs")
    ap.add_argument("--model", default=os.environ.get("FFCOACH_MODEL") or None,
                    help="use one model for everything instead of routing")
    ap.add_argument("-v", "--verbose", action="store_true", help="show routing and tool calls")
    args = ap.parse_args()

    try:
        import anthropic
    except ImportError:
        sys.exit("Missing dependency: pip install anthropic")
    fc._load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in .env (console.anthropic.com → API keys).")
    client = anthropic.Anthropic()
    d = LeagueData(args.demo)

    def answer(q):
        try:
            return ask(client, *route(client, q, history, args.model, args.verbose), d, history, q,
                       args.sms, args.verbose)
        except anthropic.APIStatusError as e:
            return f"Claude API error ({e.status_code}): {e.message}"

    history = load_session(args.session)
    if args.question:
        print(answer(" ".join(args.question)))
        save_session(args.session, history)
        return

    mode = args.model or "Sonnet for lookups, Opus for strategy"
    print(f"{d.me.name} · week {d.lg.week} · {mode} · data from {d.loaded_at}. "
          "Ask away (blank line or Ctrl-C to quit).")
    try:
        while True:
            q = input("\n> ").strip()
            if not q:
                break
            print("\n" + answer(q))
    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == "__main__":
    main()
