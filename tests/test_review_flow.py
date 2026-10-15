"""Offline checks for routing, interrupts, audit events, and posting decisions."""

import asyncio
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.github import PullRequest
from common.schemas import PRAnalysis
from exercises import exercise_4_audit as review


class FakeLLM:
    def __init__(self, initial, refined=None):
        self.initial = initial
        self.refined = refined or initial

    def with_structured_output(self, _schema):
        return self

    async def ainvoke(self, messages):
        return self.refined if "Original analysis:" in messages[-1]["content"] else self.initial


def analysis(confidence):
    return PRAnalysis(
        summary="Review summary", risk_factors=["Risk"], comments=[],
        confidence=confidence, confidence_reasoning="Test confidence",
        escalation_questions=["Why is this change needed?"],
    )


class ReviewFlowTests(unittest.TestCase):
    def test_all_routes_and_no_duplicate_pending_audit(self):
        async def check():
            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "audit.db")
                with patch.dict(os.environ, {"HITL_DB_PATH": path, "GITHUB_USER": "reviewer"}):
                    for label, initial, refined in [
                        ("auto", analysis(0.9), None),
                        ("approval", analysis(0.65), None),
                        ("edit", analysis(0.65), None),
                        ("escalation", analysis(0.4), analysis(0.68)),
                    ]:
                        posts = []
                        pr = PullRequest(
                            url="https://github.com/owner/repo/pull/1", owner="owner",
                            repo="repo", number=1, title="Test PR", author="author",
                            base_ref="main", head_ref="feature", head_sha="abcdef123",
                            diff="diff --git a/file b/file", files_changed=["file"],
                        )
                        with patch.object(review, "fetch_pr", return_value=pr), \
                             patch.object(review, "get_llm", return_value=FakeLLM(initial, refined)), \
                             patch.object(review, "post_review_comment", side_effect=lambda *args: posts.append(args)):
                            async with AsyncSqliteSaver.from_conn_string(path) as cp:
                                await cp.setup()
                                graph = review.build_graph(cp)
                                thread_id = label
                                cfg = {"configurable": {"thread_id": thread_id}}
                                result = await graph.ainvoke({"pr_url": pr.url, "thread_id": thread_id}, cfg)
                                if label == "approval":
                                    self.assertEqual(result["__interrupt__"][0].value["kind"], "approval_request")
                                    result = await graph.ainvoke(Command(resume={"choice": "reject"}), cfg)
                                elif label == "edit":
                                    result = await graph.ainvoke(Command(resume={
                                        "choice": "edit", "edited_body": "Reviewer edited comment",
                                    }), cfg)
                                elif label == "escalation":
                                    self.assertEqual(result["__interrupt__"][0].value["kind"], "escalation")
                                    result = await graph.ainvoke(Command(resume={"Why is this change needed?": "Business need"}), cfg)
                                    self.assertEqual(result["__interrupt__"][0].value["kind"], "approval_request")
                                    self.assertEqual(posts, [])
                                    result = await graph.ainvoke(Command(resume={"choice": "approve"}), cfg)
                                self.assertNotIn("__interrupt__", result)
                        self.assertEqual(len(posts), 0 if label == "approval" else 1)
                        if label == "edit":
                            self.assertEqual(posts[0][1], "Reviewer edited comment")
                    with sqlite3.connect(path) as conn:
                        rows = conn.execute(
                            "SELECT thread_id, action, decision FROM audit_events ORDER BY id"
                        ).fetchall()
                    self.assertEqual(sum(r == ("approval", "human_approval", "pending") for r in rows), 1)
                    self.assertEqual(sum(r == ("escalation", "escalate", "pending") for r in rows), 1)
                    self.assertIn(("escalation", "human_approval", "approve"), rows)
                    self.assertIn(("auto", "auto_approve", "auto"), rows)

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
