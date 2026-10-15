"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from common.github import parse_pr_url
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None
if "loaded_thread" not in st.session_state:
    st.session_state.loaded_thread = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")
    async def recent_sessions():
        async with db_conn() as conn:
            async with conn.execute("""
                SELECT thread_id, pr_url, MAX(timestamp) AS last_event,
                       CASE WHEN SUM(risk_level='high') > 0 THEN 'high'
                            WHEN SUM(risk_level='med') > 0 THEN 'med'
                            ELSE 'low' END AS worst_risk
                FROM audit_events GROUP BY thread_id, pr_url
                ORDER BY MAX(id) DESC LIMIT 25
            """) as cur:
                return [dict(row) for row in await cur.fetchall()]

    for session in asyncio.run(recent_sessions()):
        label = f"{session['worst_risk'].upper()} · {session['pr_url'].rsplit('/', 1)[-1]} · {session['last_event'][:16]}"
        if st.button(label, key=f"session_{session['thread_id']}", help=session['thread_id']):
            st.query_params["thread"] = session["thread_id"]
            st.session_state.loaded_thread = None
            st.rerun()


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key="approval_feedback")
    edited_body = st.text_area(
        "Edited comment body (required if choosing Edit)",
        key="edited_comment_body", height=180,
    )
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit"):
        if not edited_body.strip():
            st.error("Enter the edited comment body before choosing Edit.")
            return None
        return {"choice": "edit", "feedback": feedback, "edited_body": edited_body.strip()}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        answers = {
            question: st.text_area(question, key=f"question_{index}")
            for index, question in enumerate(payload["questions"])
        }
        submitted_answers = st.form_submit_button("Submit answers")
    if submitted_answers:
        if any(not answer.strip() for answer in answers.values()):
            st.error("Please answer every question.")
        else:
            return {question: answer.strip() for question, answer in answers.items()}
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        return await app.ainvoke(Command(resume=resume_value), cfg)


async def load_thread(thread_id: str):
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        snapshot = await app.aget_state(cfg)
        if snapshot.values and snapshot.next and not any(task.interrupts for task in snapshot.tasks):
            await app.ainvoke(None, cfg)
            snapshot = await app.aget_state(cfg)
        payload = next(
            (intr.value for task in snapshot.tasks for intr in task.interrupts), None
        )
        return snapshot.values, payload


requested_thread = st.query_params.get("thread")
if requested_thread and requested_thread != st.session_state.loaded_thread:
    values, payload = asyncio.run(load_thread(requested_thread))
    if values:
        st.session_state.thread_id = requested_thread
        st.session_state.pr_url = values.get("pr_url", "")
        st.session_state.interrupt_payload = payload
        st.session_state.final = values if values.get("final_action") else None
        st.session_state.loaded_thread = requested_thread
    else:
        st.error("Session not found.")


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    try:
        parse_pr_url(pr_url)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.query_params["thread"] = st.session_state.thread_id
    st.session_state.loaded_thread = st.session_state.thread_id
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR + asking the LLM..."):
        try:
            result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))
        except Exception as exc:
            st.error(f"Review failed: {exc}")
            st.stop()

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            try:
                result = asyncio.run(run_graph(
                    st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
                ))
            except Exception as exc:
                st.error(f"Resume failed: {exc}")
                st.stop()
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if action in {"auto_committed", "committed"}:
        st.success(f"✓ {action} — comment posted")
        st.link_button("View comment on GitHub", final.get("posted_comment_url") or st.session_state.pr_url)
        if final.get("analysis"):
            st.metric("Review confidence", f"{final['analysis'].confidence:.0%}")
            st.write(final["analysis"].summary)
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    elif action in {"commit_failed", "auto_commit_failed"}:
        st.error("Could not post the comment. Check GITHUB_TOKEN and GitHub permissions.")
    else:
        st.info(f"final_action = {action}")
    st.caption(f"thread_id = {st.session_state.thread_id}  ·  replay: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")
