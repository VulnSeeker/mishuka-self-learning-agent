"""
Streamlit dashboard for Onyx.

Run with:
    onyx web
or:
    streamlit run src/onyx/web.py

Pages:
    - Learn      : bootstrap a new skill
    - Run Task   : execute a task against a skill
    - Skills     : browse, inspect, delete
    - Train      : plan + train a small model
    - Settings   : view configuration (read-only)
"""

from __future__ import annotations

import json
from typing import Any, Optional

import streamlit as st

from onyx.agent import AgentError, Onyx, SkillNotFoundError
from onyx.config import CONFIG
from onyx.version import __version__


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Onyx — Self-Learning Agent",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Session-scoped agent (cached so we don't rebuild every rerun)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_agent() -> Onyx:
    return Onyx()


def _agent() -> Onyx:
    return _get_agent()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"### ◼ Onyx\n`v{__version__}`")
    st.caption(CONFIG.summary())
    st.divider()
    page = st.radio(
        "Navigate",
        ["Learn", "Run Task", "Skills", "Train", "Settings"],
        index=0,
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("[GitHub](https://github.com/vulnseeker/onyx-self-learning-agent)")


# ---------------------------------------------------------------------------
# Page: Learn
# ---------------------------------------------------------------------------

def page_learn() -> None:
    st.header("Learn a new skill")
    st.caption("Onyx will research the web and build a knowledge base for this skill.")

    with st.form("learn_form"):
        skill_name = st.text_input(
            "Skill name",
            value="OSINT",
            placeholder="e.g. OSINT, Python asyncio, Video editing",
        )
        submitted = st.form_submit_button("Start learning", type="primary")

    if not submitted:
        return

    skill_name = (skill_name or "").strip()
    if not skill_name:
        st.error("Please enter a skill name.")
        return

    status = st.status(f"Learning '{skill_name}'...", expanded=True)
    log_box = status.empty()
    lines: list[str] = []

    def _cb(msg: str) -> None:
        lines.append(msg)
        log_box.code("\n".join(lines[-30:]), language="text")

    try:
        result = _agent().learn(skill_name, progress=_cb)
        status.update(label="✅ Skill learned", state="complete")
    except AgentError as e:
        status.update(label="❌ Failed", state="error")
        st.exception(e)
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Added", result["added"])
    c2.metric("Updated", result["updated"])
    c3.metric("Skipped", result["skipped"])
    c4.metric("Total", result["total"])

    with st.expander("Skill spec", expanded=False):
        st.json(result["spec"])


# ---------------------------------------------------------------------------
# Page: Run Task
# ---------------------------------------------------------------------------

def page_run() -> None:
    st.header("Run a task")
    st.caption("Uses a learned skill's knowledge base to answer.")

    skills = _agent().skills()
    if not skills:
        st.info("No skills yet. Go to **Learn** first.")
        return

    options = {s.skill_id: f"{s.name} ({s.entry_count} entries)" for s in skills}
    skill_id = st.selectbox(
        "Skill",
        options=list(options.keys()),
        format_func=lambda x: options[x],
    )

    task = st.text_area(
        "Task",
        value="How do I enumerate email addresses for a domain?",
        height=120,
    )
    col1, col2 = st.columns([1, 1])
    show_code = col1.checkbox("Show code output", value=False)
    show_plan = col2.checkbox("Show plan", value=False)

    if not st.button("Run task", type="primary"):
        return

    task = (task or "").strip()
    if not task:
        st.error("Please enter a task.")
        return

    status = st.status("Running...", expanded=True)
    log_box = status.empty()
    lines: list[str] = []

    def _cb(msg: str) -> None:
        lines.append(msg)
        log_box.code("\n".join(lines[-30:]), language="text")

    try:
        result = _agent().run(skill_id, task, progress=_cb)
        status.update(label="✅ Done", state="complete")
    except SkillNotFoundError as e:
        status.update(label="❌ Skill not found", state="error")
        st.error(str(e))
        return
    except AgentError as e:
        status.update(label="❌ Failed", state="error")
        st.exception(e)
        return

    st.markdown("### Answer")
    st.markdown(result.answer)

    c1, c2, c3 = st.columns(3)
    c1.metric("Context entries", result.context_used)
    c2.metric("New entries added", result.new_entries)
    c3.metric("Gap detected", "yes" if result.gap.get("gap") else "no")

    if show_code and result.code_output:
        with st.expander("Code output", expanded=True):
            st.code(result.code_output, language="text")

    if show_plan and result.plan:
        with st.expander("Plan", expanded=True):
            st.json(result.plan)

    if result.gap.get("gap"):
        with st.expander("Gap details", expanded=False):
            st.json(result.gap)


# ---------------------------------------------------------------------------
# Page: Skills
# ---------------------------------------------------------------------------

def page_skills() -> None:
    st.header("Learned skills")
    skills = _agent().skills()

    if not skills:
        st.info("No skills yet. Go to **Learn** first.")
        return

    for s in skills:
        with st.expander(f"**{s.name}** — {s.entry_count} entries  ·  `{s.skill_id}`"):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.write(f"**Description:** {s.description or '(none)'}")
                st.write(f"**Tools:** {', '.join(s.tools) or '(none)'}")
                st.write(f"**Created:** {s.created_at}")
                st.write(f"**Updated:** {s.updated_at}")
            with c2:
                if st.button("Delete", key=f"del_{s.skill_id}"):
                    _agent().delete(s.skill_id)
                    st.rerun()

            if st.checkbox("Show recent knowledge", key=f"show_{s.skill_id}"):
                entries = _agent().skill_entries(s.skill_id, limit=50)
                if not entries:
                    st.caption("No entries.")
                else:
                    st.dataframe(
                        [
                            {
                                "type": e.get("type"),
                                "title": e.get("title"),
                                "confidence": e.get("confidence"),
                                "source": e.get("source_url"),
                            }
                            for e in entries
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )


# ---------------------------------------------------------------------------
# Page: Train
# ---------------------------------------------------------------------------

def page_train() -> None:
    st.header("Train a small model")
    st.caption("Onyx plans a self-contained ML experiment and runs it in the sandbox.")

    task = st.text_area(
        "Model task",
        value="Train a text classifier to detect toxic comments",
        height=100,
    )
    show_output = st.checkbox("Show sandbox output", value=False)

    if not st.button("Plan & Train", type="primary"):
        return

    task = (task or "").strip()
    if not task:
        st.error("Please enter a task.")
        return

    status = st.status("Planning and training...", expanded=True)
    log_box = status.empty()
    lines: list[str] = []

    def _cb(msg: str) -> None:
        lines.append(msg)
        log_box.code("\n".join(lines[-30:]), language="text")

    try:
        out = _agent().train_model(task, progress=_cb)
        status.update(label="✅ Done", state="complete")
    except AgentError as e:
        status.update(label="❌ Failed", state="error")
        st.exception(e)
        return

    plan = out.get("plan") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Library", plan.get("library", "?"))
    c2.metric("Dataset", plan.get("dataset_name", "?")[:30] or "?")
    c3.metric("Metric", out.get("metric") or "—")

    st.write(f"**Approach:** {plan.get('approach', '')}")
    if plan.get("dataset_url"):
        st.write(f"**Dataset URL:** {plan['dataset_url']}")

    with st.expander("Training code", expanded=False):
        st.code(plan.get("train_code", ""), language="python")

    if show_output:
        with st.expander("Sandbox output", expanded=True):
            st.code(out.get("stdout", ""), language="text")


# ---------------------------------------------------------------------------
# Page: Settings
# ---------------------------------------------------------------------------

def page_settings() -> None:
    st.header("Settings")
    st.caption("Read-only view of the current runtime configuration.")

    st.subheader("LLM")
    st.write(f"**Model:** `{CONFIG.llm_model}`")
    st.write(f"**Embedding model:** `{CONFIG.embed_model}`")
    st.write(f"**Base URL:** `{CONFIG.openai_base_url}`")
    st.write(f"**API key:** {'set' if CONFIG.openai_api_key and CONFIG.openai_api_key != 'ollama' else 'ollama (local)'}")

    st.subheader("Search")
    st.write(f"**Tavily:** {'set' if CONFIG.tavily_api_key else 'not set — using DuckDuckGo'}")

    st.subheader("Paths")
    st.write(f"**Data dir:** `{CONFIG.data_dir}`")
    st.write(f"**Skills dir:** `{CONFIG.skills_dir}`")
    st.write(f"**Chroma dir:** `{CONFIG.chroma_dir}`")
    st.write(f"**Registry:** `{CONFIG.registry_db}`")

    st.subheader("Behaviour")
    st.json({
        "max_sources_per_skill": CONFIG.max_sources_per_skill,
        "max_pages_per_source": CONFIG.max_pages_per_source,
        "request_timeout": CONFIG.request_timeout,
        "llm_max_attempts": CONFIG.llm_max_attempts,
        "code_timeout": CONFIG.code_timeout,
        "dedupe_distance": CONFIG.dedupe_distance,
        "max_chars_per_page": CONFIG.max_chars_per_page,
        "log_level": CONFIG.log_level,
    })


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_PAGES = {
    "Learn": page_learn,
    "Run Task": page_run,
    "Skills": page_skills,
    "Train": page_train,
    "Settings": page_settings,
}

_PAGES[page]()
