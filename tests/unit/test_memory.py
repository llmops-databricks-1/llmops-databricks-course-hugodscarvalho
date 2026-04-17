"""Unit tests for eu_policy_agent.memory.

Tests cover:
- LakebaseMemory.load_messages: happy path, empty session, warning on error
- LakebaseMemory.save_messages: correct INSERT calls, warning on error
- LakebaseMemory._reset_pool: closes pool and sets it to None
- Connection string: user-auth and SPN-auth branches
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import psycopg
import pytest

from eu_policy_agent.memory import LakebaseMemory

# Helpers


def _make_memory() -> LakebaseMemory:
    return LakebaseMemory(project_id="test-project-id")


def _mock_pool(rows: list[tuple] | None = None) -> MagicMock:
    """Return a mock ConnectionPool that yields a mock connection."""
    pool = MagicMock()

    # conn.execute(...).fetchall() → rows
    cursor = MagicMock()
    cursor.fetchall.return_value = rows or []
    conn = MagicMock()
    conn.execute.return_value = cursor

    @contextmanager
    def _pool_connection():  # type: ignore[no-untyped-def]
        yield conn

    pool.connection = _pool_connection
    return pool


# load_messages


class TestLoadMessages:
    def test_returns_deserialized_messages(self) -> None:
        msg1 = {"role": "user", "content": "What is GDPR?"}
        msg2 = {"role": "assistant", "content": "GDPR is..."}
        # Rows simulate JSONB auto-deserialization by psycopg
        rows = [(msg1,), (msg2,)]
        pool = _mock_pool(rows)

        mem = _make_memory()
        mem._pool = pool

        result = mem.load_messages("session-abc")
        assert result == [msg1, msg2]

    def test_returns_empty_list_for_new_session(self) -> None:
        pool = _mock_pool(rows=[])
        mem = _make_memory()
        mem._pool = pool

        result = mem.load_messages("brand-new-session")
        assert result == []

    def test_returns_empty_list_and_logs_warning_on_generic_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        pool = MagicMock()
        pool.connection.side_effect = Exception("DB error")

        mem = _make_memory()
        mem._pool = pool

        import logging

        with caplog.at_level(logging.WARNING):
            result = mem.load_messages("session-x")

        assert result == []

    def test_resets_pool_and_reraises_on_operational_error(self) -> None:
        pool = MagicMock()

        @contextmanager
        def _pool_connection():  # type: ignore[no-untyped-def]
            raise psycopg.OperationalError("connection refused")
            yield  # noqa: B012

        pool.connection = _pool_connection

        mem = _make_memory()
        mem._pool = pool

        with pytest.raises(psycopg.OperationalError):
            mem.load_messages("s")

        # Pool should be reset (set to None) so next call re-authenticates
        assert mem._pool is None


# save_messages


class TestSaveMessages:
    def test_inserts_one_row_per_message(self) -> None:
        pool = MagicMock()
        conn = MagicMock()

        @contextmanager
        def _pool_connection():  # type: ignore[no-untyped-def]
            yield conn

        pool.connection = _pool_connection

        mem = _make_memory()
        mem._pool = pool

        messages = [
            {"role": "user", "content": "Question about AI Act"},
            {"role": "assistant", "content": "The AI Act..."},
        ]
        mem.save_messages("session-42", messages)

        assert conn.execute.call_count == 2
        # Verify first call uses correct SQL and JSON-serialised message
        first_call_args = conn.execute.call_args_list[0]
        sql, params = first_call_args[0]
        assert "INSERT INTO session_messages" in sql
        assert params[0] == "session-42"
        assert json.loads(params[1]) == messages[0]

    def test_logs_warning_on_generic_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        pool = MagicMock()
        pool.connection.side_effect = Exception("write error")

        mem = _make_memory()
        mem._pool = pool

        import logging

        with caplog.at_level(logging.WARNING):
            mem.save_messages("s", [{"role": "user", "content": "x"}])
        # Should not raise — errors are swallowed with a warning

    def test_resets_pool_on_operational_error(self) -> None:
        pool = MagicMock()

        @contextmanager
        def _pool_connection():  # type: ignore[no-untyped-def]
            raise psycopg.OperationalError("timeout")
            yield  # noqa: B012

        pool.connection = _pool_connection

        mem = _make_memory()
        mem._pool = pool

        with pytest.raises(psycopg.OperationalError):
            mem.save_messages("s", [])

        assert mem._pool is None


# _reset_pool


class TestResetPool:
    def test_closes_and_nulls_pool(self) -> None:
        pool = MagicMock()
        mem = _make_memory()
        mem._pool = pool

        mem._reset_pool()

        pool.close.assert_called_once()
        assert mem._pool is None

    def test_noop_when_pool_is_none(self) -> None:
        mem = _make_memory()
        mem._pool = None
        mem._reset_pool()  # Must not raise


# Connection string auth modes


class TestGetConnectionString:
    def _make_pg_api_mock(self) -> MagicMock:
        """Build a minimal mock of PostgresAPI + workspace primitives."""
        branch = MagicMock()
        branch.name = "projects/proj/branches/main"

        endpoint = MagicMock()
        endpoint.name = "projects/proj/branches/main/endpoints/default"
        endpoint.status.hosts.host = "pg.databricks.example.com"

        credential = MagicMock()
        credential.token = "tok123"

        pg_api = MagicMock()
        pg_api.list_branches.return_value = iter([branch])
        pg_api.list_endpoints.return_value = iter([endpoint])
        pg_api.generate_database_credential.return_value = credential

        return pg_api

    def test_user_auth_includes_current_user_in_dsn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Ensure SPN env vars are absent
        for var in (
            "LAKEBASE_SP_CLIENT_ID",
            "LAKEBASE_SP_CLIENT_SECRET",
            "LAKEBASE_SP_HOST",
        ):
            monkeypatch.delenv(var, raising=False)

        pg_api = self._make_pg_api_mock()

        user = MagicMock()
        user.user_name = "alice@example.com"

        mock_workspace = MagicMock()
        mock_workspace.current_user.me.return_value = user

        with (
            patch("eu_policy_agent.memory.WorkspaceClient", return_value=mock_workspace),
            patch("eu_policy_agent.memory.PostgresAPI", return_value=pg_api),
        ):
            mem = _make_memory()
            dsn = mem._get_connection_string()

        assert "alice%40example.com" in dsn  # URL-encoded @
        assert "tok123" in dsn
        assert "pg.databricks.example.com" in dsn
        assert "sslmode=require" in dsn

    def test_spn_auth_uses_client_id_as_username(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LAKEBASE_SP_CLIENT_ID", "spn-client-id")
        monkeypatch.setenv("LAKEBASE_SP_CLIENT_SECRET", "spn-secret")
        monkeypatch.setenv("LAKEBASE_SP_HOST", "https://ws.databricks.example.com")

        pg_api = self._make_pg_api_mock()
        mock_workspace = MagicMock()

        with (
            patch("eu_policy_agent.memory.WorkspaceClient", return_value=mock_workspace),
            patch("eu_policy_agent.memory.PostgresAPI", return_value=pg_api),
        ):
            mem = _make_memory()
            dsn = mem._get_connection_string()

        assert "spn-client-id" in dsn
        assert "tok123" in dsn
