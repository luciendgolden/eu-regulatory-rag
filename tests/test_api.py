"""Tests for the FastAPI application."""

from fastapi.testclient import TestClient

from api.main import app, get_retriever


class FakeRetriever:
    def retrieve(self, query: str, regulation=None, section_type=None, top_k: int = 5):
        assert query == "What does DORA require?"
        assert regulation == "DORA"
        assert section_type == "article"
        assert top_k == 3
        return [
            {
                "score": 0.91,
                "regulation": "DORA",
                "celex": "32022R2554",
                "section_type": "article",
                "section_number": "6",
                "section_title": "ICT risk management framework",
                "chapter": "II",
                "topics": ["ict_risk"],
                "text": "Financial entities shall maintain an ICT risk management framework.",
            }
        ]

    def build_context(self, results):
        assert len(results) == 1
        return "[1] According to DORA — Article 6: Financial entities shall maintain an ICT risk management framework."


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "eu-regulatory-rag"
    assert payload["endpoints"]["retrieve"] == "/retrieve"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_retrieve_endpoint_uses_dependency_override():
    app.dependency_overrides[get_retriever] = lambda: FakeRetriever()
    try:
        response = client.post(
            "/retrieve",
            json={
                "query": "What does DORA require?",
                "regulation": "DORA",
                "section_type": "article",
                "top_k": 3,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "What does DORA require?"
    assert len(payload["results"]) == 1
    assert payload["results"][0]["section_number"] == "6"
    assert payload["context"].startswith("[1] According to DORA")
