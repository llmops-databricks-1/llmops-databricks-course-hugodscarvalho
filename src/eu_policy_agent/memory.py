"""Session memory backed by Databricks Lakebase (managed PostgreSQL).

Lakebase is a fully managed PostgreSQL-compatible database on Databricks.
This module provides ``LakebaseMemory``, a thin persistence layer that stores
and retrieves per-session conversation messages in a ``session_messages`` table.

Authentication modes
--------------------
* **User (development / notebook)**: Falls back to the ambient ``WorkspaceClient``
  identity when the SPN environment variables are absent.
* **Service Principal (production / model serving)**: Reads
  ``LAKEBASE_SP_CLIENT_ID``, ``LAKEBASE_SP_CLIENT_SECRET``, and
  ``LAKEBASE_SP_HOST`` from the environment.  Using separate env vars avoids
  overriding the default workspace client used elsewhere in the agent.

Table schema (DDL)
------------------
    CREATE TABLE IF NOT EXISTS session_messages (
        id         SERIAL PRIMARY KEY,
        session_id TEXT      NOT NULL,
        message_data JSONB   NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_session_messages_session_id
        ON session_messages (session_id);

This DDL is managed by the notebook ``3.2_session_memory_lakebase.py``;
this module only performs reads and appends.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from typing import Any

import psycopg
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import PostgresAPI
from loguru import logger
from psycopg_pool import ConnectionPool


class LakebaseMemory:
    """Read and append session messages in a Lakebase PostgreSQL database.

    Each instance lazily opens a connection pool on first use and reuses it
    across calls.  The pool is reset on ``psycopg.OperationalError`` (e.g.
    expired token) so the next operation transparently re-authenticates.

    Args:
        project_id: The Lakebase project ID (as configured in Databricks).
    """

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self._pool: ConnectionPool | None = None

    # Connection management

    def _get_connection_string(self) -> str:
        """Build a PostgreSQL connection string for the Lakebase project.

        Detects the authentication mode from environment variables:
        - SPN vars present → service-principal auth (production)
        - SPN vars absent  → current-user auth (development / notebooks)

        Returns:
            A ``postgresql://`` DSN string with ``sslmode=require``.
        """
        client_id = os.environ.get("LAKEBASE_SP_CLIENT_ID")
        client_secret = os.environ.get("LAKEBASE_SP_CLIENT_SECRET")
        host = os.environ.get("LAKEBASE_SP_HOST")

        if client_id and client_secret and host:
            w = WorkspaceClient(
                host=host,
                client_id=client_id,
                client_secret=client_secret,
            )
            username = client_id
        else:
            w = WorkspaceClient()
            user = w.current_user.me()
            username = urllib.parse.quote_plus(user.user_name)

        pg_api = PostgresAPI(w.api_client)
        project_parent = f"projects/{self.project_id}"
        default_branch = next(iter(pg_api.list_branches(parent=project_parent)))
        endpoint = next(iter(pg_api.list_endpoints(parent=default_branch.name)))
        pg_host = endpoint.status.hosts.host
        credential = pg_api.generate_database_credential(endpoint=endpoint.name)

        return (
            f"postgresql://{username}:{credential.token}@{pg_host}:5432/"
            "databricks_postgres?sslmode=require"
        )

    def _get_pool(self) -> ConnectionPool:
        """Return (creating if necessary) the connection pool."""
        if self._pool is None:
            conn_string = self._get_connection_string()
            self._pool = ConnectionPool(conninfo=conn_string, min_size=1, max_size=5)
        return self._pool

    def _reset_pool(self) -> None:
        """Close and discard the pool so the next call re-authenticates."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    # Public interface

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Load all messages for ``session_id`` in chronological order.

        Returns an empty list if no messages exist or on non-operational errors
        (logged as warnings) so the agent degrades gracefully.

        Args:
            session_id: Opaque session identifier.

        Returns:
            List of message dicts (e.g. ``{"role": "user", "content": "..."}``)
            ordered by ``created_at`` ascending.
        """
        try:
            with self._get_pool().connection() as conn:
                rows = conn.execute(
                    """
                    SELECT message_data
                    FROM session_messages
                    WHERE session_id = %s
                    ORDER BY created_at ASC
                    """,
                    (session_id,),
                ).fetchall()
                return [row[0] for row in rows]
        except psycopg.OperationalError:
            self._reset_pool()
            raise
        except Exception as exc:
            logger.warning(f"Failed to load session messages for {session_id!r}: {exc}")
            return []

    def save_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Append ``messages`` to the session record.

        Each message is stored as a separate row so ordering is preserved by
        the auto-increment ``id`` / ``created_at`` columns.

        Args:
            session_id: Opaque session identifier.
            messages: List of message dicts to persist.
        """
        try:
            with self._get_pool().connection() as conn:
                for msg in messages:
                    conn.execute(
                        "INSERT INTO session_messages (session_id, message_data) "
                        "VALUES (%s, %s)",
                        (session_id, json.dumps(msg)),
                    )
        except psycopg.OperationalError:
            self._reset_pool()
            raise
        except Exception as exc:
            logger.warning(f"Failed to save session messages for {session_id!r}: {exc}")
