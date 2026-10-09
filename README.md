# ffcoach

A weekly assistant for ESPN fantasy football. It reads every roster in your league and tells you:

- **Matchup:** your optimal projection vs. your opponent's, rough win odds, and their injured or bye-week starters.
- **Start/sit:** the best lineup for this week vs. what you have set, plus questionable players to re-check before lock.
- **Waivers:** a stacked add/drop plan (each move assumes the one above went through), plus an automatic fill for any empty slot.
- **Trades:** 1-for-1, 2-for-1 and 1-for-2 deals where **both** starting lineups improve, with a one-line pitch for the other manager.

It also comes with **ffchat**, a chat where you ask about your league in plain English. Each question is routed to the right model: quick lookups ("is Jefferson playing?") go to Claude Sonnet 5.5, and strategy questions (trades, pickups, start/sit) go to Claude Opus 5.5. See [Ask it questions](#ask-it-questions-ffchat).

## Setup (5 minutes)

```bash
pip install espn-api            # plus `anthropic` if you want --narrate
cp .env.example .env
```

Fill in `.env`:

| Variable | Where to find it |
|---|---|
| `ESPN_LEAGUE_ID` | Your league URL: `fantasy.espn.com/football/league?leagueId=`**`12345`** |
| `ESPN_TEAM_ID` | Click your team; the URL has `teamId=`**`3`** |
| `ESPN_S2`, `ESPN_SWID` | Private leagues only. Log in at fantasy.espn.com, open DevTools → Application → Cookies → `espn.com`, copy `espn_s2` and `SWID` (keep the braces in SWID). |

The cookies are your login. Keep `.env` out of git. They last about a year.

## Run

```bash
python ffcoach.py --demo          # try it on a fake league first
python ffcoach.py                 # full report for your league
python ffcoach.py --sms           # 6-line version for a text message
python ffcoach.py --narrate       # adds a short LLM-written brief (needs ANTHROPIC_API_KEY)
python ffcoach.py --json out.json # raw analysis, for logging or backtesting
```

## Ask it questions (ffchat)

`ffchat.py` lets you ask about your league in plain English. Claude reads your live rosters, matchups and free agents through a few tools and does the reasoning itself. The one piece of math left in code is trade scoring (`evaluate_trades`), because re-solving both teams' best lineups is easy to get wrong by hand.

**It models the other managers.** ffchat reads your league's full transaction history from ESPN: every waiver bid (including losing bids and amounts), free-agent pickups, completed trades, trade offers made and received, accept/decline counts, lineup activity, FAAB left, waiver priority and draft picks. With that it can:
- rank trade ideas by how likely *that specific manager* is to accept, not just by value;
- predict who'll compete for your waiver targets and suggest FAAB bids based on what your league actually pays;
- flag managers who are inactive and unlikely to answer an offer at all.

ESPN shows some trade offers between other teams but doesn't always link them to their outcome, so acceptance odds lean on each manager's overall trading pattern.

It also sees the season so far: standings with points for/against and ESPN playoff odds, every past matchup, each player's weekly points vs projection, and remaining schedules. For news it uses Claude's web search (up to 10 searches per question) to check injury, practice and roster reports.

Every question and answer is saved to `logs/<name>.md` (git-ignored). The terminal chat uses `FFCOACH_USER` from `.env` as the name; the web app uses each person's login.

Each question is routed by a quick Haiku check: lookups go to Claude Sonnet 5.5, strategy (trades, pickups, start/sit) to Claude Opus 5.5.

```bash
pip install anthropic            # and put ANTHROPIC_API_KEY in .env
python3 ffchat.py                # interactive chat
python3 ffchat.py "should I start Puka or Waddle this week?"
python3 ffchat.py -v "find me a trade for a better RB"   # -v shows routing and tool calls
python3 ffchat.py "which trades is each manager most likely to accept?"
python3 ffchat.py "who else will bid on the RB I want, and how much should I bid?"
python3 ffchat.py --model claude-opus-5-5                # skip routing, use one model
```

Optional `.env` settings: `FFCOACH_EFFORT` (Opus, default `high`) and `FFCOACH_SIMPLE_EFFORT` (Sonnet, default `medium`).

What it can't know: breaking news, practice reports and weather. It only sees what ESPN's data has when you ask.

## Web app (share with friends in your league)

`app.py` puts ffchat behind a password-protected web page. Everyone signs in with their own name and password, gets advice for their own team, and is limited to a few questions a day on your API key, unless they paste their own key in the sidebar.

**Deploy on Streamlit Community Cloud (free):**
1. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **Create app**.
2. Pick this repo, branch `main`, main file `app.py`.
3. Under **Advanced settings → Secrets**, paste your settings (format below) and deploy.
4. Share the app link plus each friend's name and password.

**Secrets format** (also works locally as `.streamlit/secrets.toml`, which is git-ignored):

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
ESPN_LEAGUE_ID = "12345"
ESPN_TEAM_ID = "3"            # your team; used to load the league
ESPN_YEAR = "2026"
ESPN_S2 = "..."
ESPN_SWID = "{...}"
DAILY_LIMIT = 10              # questions per person per day on your key
UNLIMITED_USERS = ["you"]     # no daily limit

[users]                       # name = password
you = "pick-a-password"
friend = "another-password"

[user_teams]                  # lock each person to their ESPN team ID; anyone not listed picks from a dropdown
you = 3
friend = 6
```

Run it locally with `streamlit run app.py`. The daily counter lives in memory, so it resets if the app restarts. Each person's conversation is private to their browser session, but anyone in the league who uses it can get trade advice aimed at you.

## Hook it into OpenClaw

Have OpenClaw run `python ffcoach.py --sms` and text you the output on two schedules:
- **Tuesday ~8pm CT**, before waivers process: waiver plan and trade ideas.
- **Thursday ~5pm CT**, before TNF lock, and **Sunday ~11am CT**: start/sit and injury checks.

For texting questions, add an OpenClaw skill that runs on incoming fantasy messages:

```bash
python /path/to/ffcoach/ffchat.py --sms --session ~/.ffchat-session.json "<the text you sent>"
```

and replies with whatever it prints. `--sms` keeps answers text-length. `--session` remembers the last few exchanges for two hours, so "what about at flex?" works as a follow-up. Use a prefix like `ff:` so OpenClaw knows which texts to route here.

## How it decides (and where it's weak)

- **Player value** blends ESPN's season-projection average (60%) with the actual average so far (40%). Change `ROS_PROJ_WEIGHT` to adjust.
- **Injuries** discount this week's projection: Questionable 85%, Doubtful 25%, Out/IR 0.
- **Trades** only count rest-of-season *starting lineup* points, so bench depth is worth zero. That's deliberate: it keeps you from trading for players you won't start. It does undervalue insurance late in the season, though.
- **Win odds** are a normal approximation on the projection gap. Treat them as directional, not exact.
- It uses ESPN's unofficial API through `espn-api`. If ESPN changes something, update the library first.

The obvious next upgrade is **backtesting**: save `--json` every week, then compare the start/sit calls against actual scores, so you know whether the tool beats your gut.
