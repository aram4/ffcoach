"""
Web version of ffchat for you and friends in the same ESPN league.

  streamlit run app.py

Settings come from Streamlit secrets (.streamlit/secrets.toml locally, the Secrets
box on Streamlit Community Cloud), falling back to .env. See README for the format.
"""
import datetime
import hmac
import os

import anthropic
import streamlit as st

import ffcoach as fc
import ffchat as ch

st.set_page_config(page_title="ffcoach", page_icon="🏈")

for k, v in st.secrets.items():
    if isinstance(v, (str, int)):
        os.environ.setdefault(k, str(v))
fc._load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

USERS = dict(st.secrets.get("users", {}))
DAILY_LIMIT = int(st.secrets.get("DAILY_LIMIT", 10))
UNLIMITED = set(st.secrets.get("UNLIMITED_USERS", []))


@st.cache_resource
def usage_counter() -> dict:
    # Shared by all sessions; resets when the app restarts, which is fine for a soft cap.
    return {}


@st.cache_resource(ttl=3600)
def team_names() -> dict[int, str]:
    lg = ch.LeagueData(False).lg
    return {tid: t.name for tid, t in sorted(lg.teams.items())}


def login():
    st.title("🏈 ffcoach")
    if not USERS:
        st.error("No users configured. Add a [users] section to the app's secrets.")
        st.stop()
    with st.form("login"):
        name = st.text_input("Name").strip().lower()
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in"):
            if name in USERS and hmac.compare_digest(pw, str(USERS[name])):
                st.session_state.user = name
                st.rerun()
            st.error("Wrong name or password.")
    st.stop()


if "user" not in st.session_state:
    login()
user = st.session_state.user

with st.sidebar:
    st.write(f"Signed in as **{user}**")
    teams = team_names()
    fixed = dict(st.secrets.get("user_teams", {})).get(user)
    if fixed:
        team_id = int(fixed)
        st.write(f"Team: **{teams.get(team_id, team_id)}**")
    else:
        team_id = st.selectbox("Your team", list(teams), format_func=teams.get, index=None,
                               placeholder="Pick your team", key="team_id")
    own_key = st.text_input("Your own Anthropic API key (optional, removes the daily limit)",
                            type="password").strip()
    if st.button("New conversation"):
        for k in ("chat", "history"):
            st.session_state.pop(k, None)
        st.rerun()

if team_id is None:
    st.title("🏈 ffcoach")
    st.info("Pick your team in the sidebar to get started.")
    st.stop()
if st.session_state.get("loaded_team") != team_id:
    with st.spinner("Loading your league from ESPN…"):
        st.session_state.league = ch.LeagueData(False, team_id)
    st.session_state.loaded_team = team_id
    st.session_state.chat, st.session_state.history = [], []
st.session_state.setdefault("chat", [])
st.session_state.setdefault("history", [])
d = st.session_state.league

today = datetime.date.today().isoformat()
counts = usage_counter()
used = counts.get((user, today), 0)
limited = not own_key and user not in UNLIMITED

st.title("🏈 ffcoach")
st.caption(f"{d.me.name} · week {d.lg.week} · data from {d.loaded_at}"
           + (f" · {max(DAILY_LIMIT - used, 0)} of {DAILY_LIMIT} questions left today" if limited else ""))

for role, text in st.session_state.chat:
    st.chat_message(role).markdown(text)

q = st.chat_input("Ask about trades, waivers, lineups…")
if q:
    if limited and used >= DAILY_LIMIT:
        st.warning(f"You've used today's {DAILY_LIMIT} questions. Add your own API key in the sidebar to keep going.")
        st.stop()
    st.chat_message("user").markdown(q)
    st.session_state.chat.append(("user", q))
    client = anthropic.Anthropic(api_key=own_key) if own_key else anthropic.Anthropic()
    with st.chat_message("assistant"):
        with st.status("Thinking… strategy questions take a minute or two.") as status:
            try:
                model, effort = ch.route(client, q, st.session_state.history, None, False)
                status.write(f"Using {model}")
                answer = ch.ask(client, model, effort, d, st.session_state.history, q, False, False,
                                log=lambda m: status.write(m.split("(")[0].replace("_", " ")))
                ch.log_exchange(user, d.me.name, model, q, answer)
            except anthropic.APIStatusError as e:
                answer = f"Claude API error ({e.status_code}): {e.message}"
            status.update(label="Done", state="complete")
        st.markdown(answer)
    st.session_state.chat.append(("assistant", answer))
    if limited:
        counts[(user, today)] = used + 1
