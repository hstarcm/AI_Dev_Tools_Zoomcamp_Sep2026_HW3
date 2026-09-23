"""Protocol tests for the SQLite starter.

These tests intentionally exercise storage calls from multiple threads: that
is the closest local equivalent to several worker processes racing to claim an
inbox.  The production guarantee comes from SQLite's BEGIN IMMEDIATE boundary,
not from a Python lock.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Default to a scratch DB so `pytest` never resets the dev server's
# `./agent-relay.db`. Respect an explicit RELAY_DATABASE_URL/DATABASE_URL
# (e.g. CI pointing at PostgreSQL), but otherwise isolate tests.
default_test_db = (Path(tempfile.gettempdir()) / "agent-relay-test.db").as_posix()
os.environ.setdefault("RELAY_DATABASE_URL", f"sqlite:///{default_test_db}")

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

import main
from database import Agent, Attempt, Base, Task, as_db_time, db_session, engine, utcnow
from storage import claim_one


@pytest.fixture(autouse=True)
def empty_database():
    # Resets whatever DB RELAY_DATABASE_URL points at. Defaults to the
    # scratch /tmp file above; never run against a DB with data you need.
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


def register(client: TestClient, name: str) -> tuple[dict, dict[str, str]]:
    response = client.post("/api/v1/agents", json={"name": name})
    assert response.status_code == 201
    data = response.json()
    return data, {"Authorization": f"Bearer {data['token']}"}


def test_protocol_idempotency_terminal_retry_and_auth_boundary():
    with TestClient(main.app) as client:
        sender, sender_headers = register(client, "sender")
        recipient, recipient_headers = register(client, "uppercase")
        sent = client.post(
            "/api/v1/tasks",
            headers={**sender_headers, "Idempotency-Key": "demo-1"},
            json={"to": recipient["agent_id"], "input": "hello relay"},
        )
        assert sent.status_code == 201
        duplicate = client.post(
            "/api/v1/tasks",
            headers={**sender_headers, "Idempotency-Key": "demo-1"},
            json={"to": recipient["agent_id"], "input": "hello relay"},
        )
        assert duplicate.status_code == 201
        assert duplicate.json() == sent.json()
        conflict = client.post(
            "/api/v1/tasks",
            headers={**sender_headers, "Idempotency-Key": "demo-1"},
            json={"to": recipient["agent_id"], "input": "different"},
        )
        assert conflict.status_code == 409

        task_id = sent.json()["task_id"]
        claim = client.post(
            "/api/v1/tasks/claim",
            headers=recipient_headers,
            json={"worker_id": "worker-a", "wait_seconds": 0},
        )
        assert claim.status_code == 200
        claim_data = claim.json()
        assert "claim_token" in claim_data
        complete = client.post(
            f"/api/v1/tasks/{task_id}/complete",
            headers=recipient_headers,
            json={"claim_token": claim_data["claim_token"], "output": "HELLO RELAY"},
        )
        assert complete.status_code == 200
        retry = client.post(
            f"/api/v1/tasks/{task_id}/complete",
            headers=recipient_headers,
            json={"claim_token": claim_data["claim_token"], "output": "HELLO RELAY"},
        )
        assert retry.status_code == 200
        assert client.get(f"/api/v1/tasks/{task_id}", headers=recipient_headers).status_code == 200
        forbidden = client.get(f"/api/v1/tasks/{task_id}", headers={"Authorization": f"Bearer {sender['token']}"})
        assert forbidden.status_code == 200  # sender is an authorized participant
        no_credentials = client.get("/api/v1/agents")
        assert no_credentials.status_code == 401
        attempts = client.get(f"/api/v1/tasks/{task_id}/attempts", headers=sender_headers).json()
        assert attempts["items"][0]["outcome"] == "completed"
        assert "claim_token" not in attempts["items"][0]


def test_sqlite_atomic_claims_distribute_without_overlap():
    with TestClient(main.app) as client:
        _sender, sender_headers = register(client, "sender")
        recipient, _recipient_headers = register(client, "recipient")
        for index in range(16):
            response = client.post(
                "/api/v1/tasks",
                headers=sender_headers,
                json={"to": recipient["agent_id"], "input": f"task-{index}"},
            )
            assert response.status_code == 201
        with ThreadPoolExecutor(max_workers=16) as pool:
            claims = list(pool.map(lambda index: claim_one(recipient["agent_id"], f"worker-{index}"), range(16)))
        claims = [claim for claim in claims if claim is not None]
        assert len(claims) == 16
        assert len({claim["task_id"] for claim in claims}) == 16
        with db_session() as db:
            processing = list(db.query(Task).filter(Task.status == "processing"))
            assert len(processing) == 16
            assert all(task.attempt_count == 1 for task in processing)


def test_expiry_requeues_and_old_token_is_stale_before_recovery():
    with TestClient(main.app) as client:
        _sender, sender_headers = register(client, "sender")
        recipient, recipient_headers = register(client, "recipient")
        task = client.post(
            "/api/v1/tasks",
            headers=sender_headers,
            json={"to": recipient["agent_id"], "input": "recover me"},
        ).json()
        task_id = task["task_id"]
        first = client.post(
            "/api/v1/tasks/claim", headers=recipient_headers, json={"worker_id": "dead", "wait_seconds": 0}
        ).json()
        with db_session() as db:
            attempt = db.query(Attempt).filter(Attempt.task_id == task_id).one()
            attempt.lease_expires_at = as_db_time(utcnow() - timedelta(seconds=1))
        stale = client.post(
            f"/api/v1/tasks/{task_id}/complete",
            headers=recipient_headers,
            json={"claim_token": first["claim_token"], "output": "TOO LATE"},
        )
        assert stale.status_code == 409
        assert stale.json()["error"]["code"] == "stale_claim"
        assert main.recover_expired() == 1
        second = client.post(
            "/api/v1/tasks/claim", headers=recipient_headers, json={"worker_id": "replacement", "wait_seconds": 0}
        )
        assert second.status_code == 200
        assert second.json()["attempt"] == 2
        assert second.json()["claim_token"] != first["claim_token"]


def test_dashboard_is_asset_and_invalid_input_is_documented_error():
    with TestClient(main.app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "sessionStorage" in page.text
        missing_name = client.post("/api/v1/agents", json={})
        assert missing_name.status_code == 400
        assert missing_name.json()["error"]["code"] == "invalid_input"


def test_acceptance_scenario_1_task_exchange_and_result():
    """Acceptance Scenario 1 (SPEC.md):
    Register two agents. One sends a task; the other claims and completes it; the sender reads the result.
    Verifies the complete lifecycle across both HTTP API and real database state.
    """
    with TestClient(main.app) as client:
        # 1. Register two agents: Alice (requester) and Bob (worker)
        alice, alice_headers = register(client, "alice")
        bob, bob_headers = register(client, "bob")
        assert alice["agent_id"] != bob["agent_id"]
        assert alice["token"].startswith("agt_")
        assert bob["token"].startswith("agt_")

        # 2. Alice sends a task to Bob
        task_input = "Translate 'Good morning' into Spanish and French"
        task_response = client.post(
            "/api/v1/tasks",
            headers=alice_headers,
            json={"to": bob["agent_id"], "input": task_input},
        )
        assert task_response.status_code == 201
        task_data = task_response.json()
        task_id = task_data["task_id"]
        assert task_data["status"] == "queued"

        # Verify task is queued in DB
        with db_session() as db:
            db_task = db.query(Task).filter(Task.id == task_id).one()
            assert db_task.status == "queued"
            assert db_task.sender_id == alice["agent_id"]
            assert db_task.recipient_id == bob["agent_id"]
            assert db_task.input == task_input
            assert db_task.output is None
            assert db_task.attempt_count == 0

        # 3. Bob claims the task
        claim_response = client.post(
            "/api/v1/tasks/claim",
            headers=bob_headers,
            json={"worker_id": "bob-worker-1", "wait_seconds": 0},
        )
        assert claim_response.status_code == 200
        claim_data = claim_response.json()
        assert claim_data["task_id"] == task_id
        assert claim_data["from"] == alice["agent_id"]
        assert claim_data["input"] == task_input
        assert claim_data["attempt"] == 1
        claim_token = claim_data["claim_token"]
        assert claim_token.startswith("clm_")

        # Verify task is processing in DB with attempt created
        with db_session() as db:
            db_task = db.query(Task).filter(Task.id == task_id).one()
            assert db_task.status == "processing"
            assert db_task.attempt_count == 1
            db_attempt = db.query(Attempt).filter(Attempt.task_id == task_id).one()
            assert db_attempt.attempt_number == 1
            assert db_attempt.worker_id == "bob-worker-1"
            assert db_attempt.outcome == "processing"
            assert db_attempt.finished_at is None

        # 4. Bob completes the task with the result
        expected_output = "Spanish: 'Buenos días', French: 'Bonjour'"
        complete_response = client.post(
            f"/api/v1/tasks/{task_id}/complete",
            headers=bob_headers,
            json={"claim_token": claim_token, "output": expected_output},
        )
        assert complete_response.status_code == 200
        assert complete_response.json()["status"] == "completed"

        # 5. Alice reads the task result
        result_response = client.get(f"/api/v1/tasks/{task_id}", headers=alice_headers)
        assert result_response.status_code == 200
        result = result_response.json()
        assert result["task_id"] == task_id
        assert result["from"] == alice["agent_id"]
        assert result["to"] == bob["agent_id"]
        assert result["status"] == "completed"
        assert result["input"] == task_input
        assert result["output"] == expected_output
        assert result["error"] is None
        assert result["attempt_count"] == 1
        assert result["finished_at"] is not None

        # 6. Verify dashboard query endpoints and attempts history
        sent_tasks = client.get("/api/v1/tasks?direction=sent", headers=alice_headers)
        assert sent_tasks.status_code == 200
        assert any(t["task_id"] == task_id for t in sent_tasks.json()["items"])

        received_tasks = client.get("/api/v1/tasks?direction=received", headers=bob_headers)
        assert received_tasks.status_code == 200
        assert any(t["task_id"] == task_id for t in received_tasks.json()["items"])

        attempts_response = client.get(f"/api/v1/tasks/{task_id}/attempts", headers=alice_headers)
        assert attempts_response.status_code == 200
        attempts_items = attempts_response.json()["items"]
        assert len(attempts_items) == 1
        assert attempts_items[0]["attempt"] == 1
        assert attempts_items[0]["outcome"] == "completed"
        assert attempts_items[0]["worker_id"] == "bob-worker-1"

        # 7. Verify final persisted state in real DB
        with db_session() as db:
            db_task = db.query(Task).filter(Task.id == task_id).one()
            assert db_task.status == "completed"
            assert db_task.output == expected_output
            assert db_task.finished_at is not None

            db_attempt = db.query(Attempt).filter(Attempt.task_id == task_id).one()
            assert db_attempt.outcome == "completed"
            assert db_attempt.finished_at is not None
            assert db_attempt.worker_id == "bob-worker-1"

            # Check both agents exist in DB with token hashes
            alice_db = db.query(Agent).filter(Agent.id == alice["agent_id"]).one()
            bob_db = db.query(Agent).filter(Agent.id == bob["agent_id"]).one()
            assert alice_db.token_hash is not None
            assert bob_db.token_hash is not None

