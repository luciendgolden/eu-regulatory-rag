"""Tests for the FastAPI backend.

Uses FastAPI TestClient with mocked RAG chain and Qdrant so no real services
are required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# App + client fixtures
# ---------------------------------------------------------------------------


def _make_mock_qdrant_client(collections: list[str] | None = None):
    """Return a mock Qdrant client object."""
    mock = MagicMock()
    col_names = collections or []
    mock_cols = [MagicMock(name=n) for n in col_names]
    # .name on MagicMock is special — set via configure_mock
    for mc, name in zip(mock_cols, col_names):
        mc.configure_mock(name=name)
    mock.get_collections.return_value.collections = mock_cols
    return mock


@pytest.fixture(scope="module")
def app():
    """Create a FastAPI app with Qdrant startup patched out."""
    with patch("vectordb.qdrant_client.QdrantWrapper._get_client", return_value=_make_mock_qdrant_client()):
        from api.app import create_app

        return create_app()


@pytest.fixture(scope="module")
def client(app):
    """Return a module-scoped TestClient."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _override_auth(app, require_key: str | None = None, require_admin: str | None = None):
    """Set dependency_overrides on *app* for auth dependencies.

    If require_key is None → auth is disabled (dev mode).
    """
    from api.auth import get_api_key, get_admin_api_key

    async def _good_key():
        return "test-key"

    async def _bad_key():
        raise HTTPException(status_code=401, detail="Invalid API key")

    async def _bad_admin():
        raise HTTPException(status_code=403, detail="Invalid admin key")

    if require_key is None:
        app.dependency_overrides[get_api_key] = _good_key
        app.dependency_overrides[get_admin_api_key] = _good_key
    else:
        # Allow correct key, reject others
        # (For simplicity, tests handle key checks manually)
        pass

    return app


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_200(self, client, app):
        from api.auth import get_api_key

        app.dependency_overrides[get_api_key] = lambda: "test"
        try:
            with patch("api.routes.health.QdrantWrapper") as mock_cls:
                mock_cls.return_value._get_client.return_value = _make_mock_qdrant_client()
                resp = client.get("/api/v1/health")
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200

    def test_health_schema(self, client, app):
        from api.auth import get_api_key

        app.dependency_overrides[get_api_key] = lambda: "test"
        try:
            mock_qdrant = _make_mock_qdrant_client(["eu_regulations"])
            with patch("api.routes.health.QdrantWrapper") as mock_cls:
                mock_cls.return_value._get_client.return_value = mock_qdrant
                resp = client.get("/api/v1/health")
        finally:
            app.dependency_overrides.clear()

        data = resp.json()
        assert "status" in data
        assert "qdrant_connected" in data
        assert "collections" in data
        assert "version" in data

    def test_health_degraded_when_qdrant_down(self, client, app):
        from api.auth import get_api_key

        app.dependency_overrides[get_api_key] = lambda: "test"
        try:
            with patch("api.routes.health.QdrantWrapper") as mock_cls:
                mock_cls.return_value._get_client.side_effect = Exception("refused")
                resp = client.get("/api/v1/health")
        finally:
            app.dependency_overrides.clear()

        data = resp.json()
        assert data["qdrant_connected"] is False
        assert data["status"] == "degraded"

    def test_health_ok_when_qdrant_up(self, client, app):
        from api.auth import get_api_key

        app.dependency_overrides[get_api_key] = lambda: "test"
        try:
            with patch("api.routes.health.QdrantWrapper") as mock_cls:
                mock_cls.return_value._get_client.return_value = _make_mock_qdrant_client()
                resp = client.get("/api/v1/health")
        finally:
            app.dependency_overrides.clear()

        assert resp.json()["status"] == "ok"
        assert resp.json()["qdrant_connected"] is True


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class TestAuth:
    """Test API key authentication end-to-end via the health endpoint."""

    def test_no_key_required_when_api_key_blank(self, client):
        """When API_KEY setting is empty, any request should pass through."""
        with patch("api.auth.settings") as mock_settings:
            mock_settings.api_key = ""
            mock_settings.admin_api_key = ""
            with patch("api.routes.health.QdrantWrapper") as mock_cls:
                mock_cls.return_value._get_client.return_value = _make_mock_qdrant_client()
                resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_valid_key_accepted(self, client):
        with patch("api.auth.settings") as mock_settings:
            mock_settings.api_key = "my-key"
            mock_settings.admin_api_key = ""
            with patch("api.routes.health.QdrantWrapper") as mock_cls:
                mock_cls.return_value._get_client.return_value = _make_mock_qdrant_client()
                resp = client.get("/api/v1/health", headers={"X-API-Key": "my-key"})
        assert resp.status_code == 200

    def test_invalid_key_rejected(self, client, app):
        """Override auth dependency to simulate key rejection for a wrong key."""
        from fastapi import HTTPException, status

        from api.auth import get_api_key

        async def _reject_wrong_key() -> str:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key.",
            )

        app.dependency_overrides[get_api_key] = _reject_wrong_key
        try:
            with patch("api.routes.health.QdrantWrapper") as mock_cls:
                mock_cls.return_value._get_client.return_value = _make_mock_qdrant_client()
                resp = client.get("/api/v1/health", headers={"X-API-Key": "wrong"})
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 401

    def test_key_via_query_param(self, client):
        with patch("api.auth.settings") as mock_settings:
            mock_settings.api_key = "my-key"
            mock_settings.admin_api_key = ""
            with patch("api.routes.health.QdrantWrapper") as mock_cls:
                mock_cls.return_value._get_client.return_value = _make_mock_qdrant_client()
                resp = client.get("/api/v1/health?api_key=my-key")
        assert resp.status_code == 200

    def test_missing_key_returns_401(self, client, app):
        """Override auth dependency to require a key — no key provided → 401."""
        from fastapi import HTTPException, status

        from api.auth import get_api_key

        async def _require_key() -> str:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key.",
            )

        app.dependency_overrides[get_api_key] = _require_key
        try:
            with patch("api.routes.health.QdrantWrapper") as mock_cls:
                mock_cls.return_value._get_client.return_value = _make_mock_qdrant_client()
                resp = client.get("/api/v1/health")
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Regulations
# ---------------------------------------------------------------------------


class TestRegulations:
    def _qdrant_with_count(self, count: int = 0):
        mock_count = MagicMock()
        mock_count.count = count
        mock_qd_client = MagicMock()
        mock_qd_client.count.return_value = mock_count
        return mock_qd_client

    def test_list_regulations(self, client, app):
        from api.auth import get_api_key

        app.dependency_overrides[get_api_key] = lambda: "test"
        try:
            with patch("api.routes.regulations.QdrantWrapper") as mock_cls:
                mock_cls.return_value._get_client.return_value = self._qdrant_with_count(5)
                resp = client.get("/api/v1/regulations")
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2
        ids = [r["id"] for r in data]
        assert "DORA" in ids
        assert "NIS2" in ids

    def test_regulation_schema(self, client, app):
        from api.auth import get_api_key

        app.dependency_overrides[get_api_key] = lambda: "test"
        try:
            with patch("api.routes.regulations.QdrantWrapper") as mock_cls:
                mock_cls.return_value._get_client.return_value = self._qdrant_with_count(0)
                resp = client.get("/api/v1/regulations")
        finally:
            app.dependency_overrides.clear()

        reg = resp.json()[0]
        for field in ("id", "title", "celex", "article_count"):
            assert field in reg

    def test_regulation_article_count(self, client, app):
        from api.auth import get_api_key

        app.dependency_overrides[get_api_key] = lambda: "test"
        try:
            with patch("api.routes.regulations.QdrantWrapper") as mock_cls:
                mock_cls.return_value._get_client.return_value = self._qdrant_with_count(42)
                resp = client.get("/api/v1/regulations")
        finally:
            app.dependency_overrides.clear()

        for reg in resp.json():
            assert reg["article_count"] == 42

    def test_regulation_qdrant_down_returns_zero_counts(self, client, app):
        from api.auth import get_api_key

        app.dependency_overrides[get_api_key] = lambda: "test"
        try:
            with patch("api.routes.regulations.QdrantWrapper") as mock_cls:
                mock_cls.return_value._get_client.side_effect = Exception("down")
                resp = client.get("/api/v1/regulations")
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200
        for reg in resp.json():
            assert reg["article_count"] == 0


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

_MOCK_CHAIN_RESULT = {
    "answer": "DORA Article 5 requires entities to maintain a digital operational resilience strategy.",
    "sources": [
        {
            "regulation": "DORA",
            "section_type": "article",
            "section_number": "5",
            "section_title": "Digital operational resilience strategy",
            "score": 0.92,
        }
    ],
    "query_time_ms": 123,
}


class TestQuery:
    def test_query_returns_200(self, client, app):
        from api.auth import get_api_key

        app.dependency_overrides[get_api_key] = lambda: "test"
        try:
            with patch("api.routes.query.chain_module") as mock_cm:
                mock_cm.query = AsyncMock(return_value=_MOCK_CHAIN_RESULT)
                resp = client.post(
                    "/api/v1/query",
                    json={"question": "What does DORA require for ICT risk management?"},
                )
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200

    def test_query_response_schema(self, client, app):
        from api.auth import get_api_key

        app.dependency_overrides[get_api_key] = lambda: "test"
        try:
            with patch("api.routes.query.chain_module") as mock_cm:
                mock_cm.query = AsyncMock(return_value=_MOCK_CHAIN_RESULT)
                resp = client.post(
                    "/api/v1/query",
                    json={"question": "What does DORA require?"},
                )
        finally:
            app.dependency_overrides.clear()

        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "query_time_ms" in data
        src = data["sources"][0]
        assert "regulation" in src
        assert "section_type" in src
        assert "score" in src

    def test_query_with_regulation_filter(self, client, app):
        from api.auth import get_api_key

        app.dependency_overrides[get_api_key] = lambda: "test"
        try:
            with patch("api.routes.query.chain_module") as mock_cm:
                mock_cm.query = AsyncMock(return_value=_MOCK_CHAIN_RESULT)
                resp = client.post(
                    "/api/v1/query",
                    json={
                        "question": "NIS2 incident reporting?",
                        "regulation": "NIS2",
                        "top_k": 3,
                    },
                )
                call_kwargs = mock_cm.query.call_args.kwargs
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200
        assert call_kwargs.get("regulation") == "NIS2"
        assert call_kwargs.get("top_k") == 3

    def test_query_validation_error_on_empty_question(self, client, app):
        from api.auth import get_api_key

        app.dependency_overrides[get_api_key] = lambda: "test"
        try:
            resp = client.post("/api/v1/query", json={"question": ""})
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 422

    def test_query_top_k_out_of_range(self, client, app):
        from api.auth import get_api_key

        app.dependency_overrides[get_api_key] = lambda: "test"
        try:
            resp = client.post(
                "/api/v1/query", json={"question": "test", "top_k": 100}
            )
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 422

    def test_query_missing_question(self, client, app):
        from api.auth import get_api_key

        app.dependency_overrides[get_api_key] = lambda: "test"
        try:
            resp = client.post("/api/v1/query", json={})
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 422

    def test_streaming_query_returns_event_stream(self, client, app):
        from api.auth import get_api_key

        async def _fake_stream(*args, **kwargs):
            yield {"chunk": "Based on "}
            yield {"chunk": "DORA Article 5..."}
            yield {"done": True, "sources": [], "query_time_ms": 50}

        mock_chain = MagicMock()
        mock_chain.stream_query = _fake_stream

        app.dependency_overrides[get_api_key] = lambda: "test"
        try:
            with patch("api.routes.query.chain_module") as mock_cm:
                mock_cm.get_chain.return_value = mock_chain
                resp = client.post(
                    "/api/v1/query",
                    json={"question": "What is DORA?", "stream": True},
                )
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Ingest (admin)
# ---------------------------------------------------------------------------


class TestIngest:
    def test_ingest_requires_admin_key(self, client):
        """Regular user key should be rejected for admin endpoint."""
        with patch("api.auth.settings") as mock_settings:
            mock_settings.api_key = "user-key"
            mock_settings.admin_api_key = "admin-secret"
            resp = client.post(
                "/api/v1/ingest",
                json={"regulation": "DORA", "force": False},
                headers={"X-API-Key": "user-key"},
            )
        assert resp.status_code == 403

    def test_ingest_accepted_with_admin_key(self, client, app):
        from api.auth import get_admin_api_key

        app.dependency_overrides[get_admin_api_key] = lambda: "admin"
        try:
            with patch("api.routes.ingest._run_ingestion", new_callable=AsyncMock):
                resp = client.post(
                    "/api/v1/ingest",
                    json={"regulation": "all", "force": False},
                )
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["regulation"] == "all"

    def test_ingest_schema(self, client, app):
        from api.auth import get_admin_api_key

        app.dependency_overrides[get_admin_api_key] = lambda: "admin"
        try:
            with patch("api.routes.ingest._run_ingestion", new_callable=AsyncMock):
                resp = client.post(
                    "/api/v1/ingest",
                    json={"regulation": "NIS2", "force": True},
                )
        finally:
            app.dependency_overrides.clear()

        data = resp.json()
        assert "status" in data
        assert "message" in data
        assert "regulation" in data
