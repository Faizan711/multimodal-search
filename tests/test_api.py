"""
tests/test_api.py
──────────────────
Integration tests for the FastAPI endpoints.

These tests require Qdrant to be running (docker run qdrant).
They use httpx's TestClient to call the API without starting a real server.

Run with:
  source .venv/bin/activate
  pytest tests/test_api.py -v

Learning notes:
  - TestClient from fastapi.testclient spins up the app in-process (no network needed).
  - We use pytest fixtures (setup code shared across tests) to avoid repetition.
  - Testing HTTP APIs: always test status codes AND response body structure.
"""

import io

import pytest
import requests
from fastapi.testclient import TestClient
from PIL import Image

from app.api import app

client = TestClient(app)


class TestHealth:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"


class TestCollectionInfo:
    def test_info_returns_200(self):
        response = client.get("/info")
        assert response.status_code == 200

    def test_info_has_expected_fields(self):
        response = client.get("/info")
        data = response.json()
        assert "points_count" in data
        assert "status" in data


class TestTextSearch:
    def test_returns_200_for_valid_query(self):
        response = client.get("/search/text", params={"q": "a cat"})
        assert response.status_code == 200

    def test_response_has_results_field(self):
        response = client.get("/search/text", params={"q": "a sunset"})
        data = response.json()
        assert "results" in data
        assert "query" in data
        assert "count" in data

    def test_top_k_respected(self):
        response = client.get("/search/text", params={"q": "a dog", "top_k": 3})
        data = response.json()
        assert data["count"] <= 3

    def test_empty_query_returns_422(self):
        response = client.get("/search/text", params={"q": ""})
        assert response.status_code == 422  # FastAPI validation error

    def test_missing_query_returns_422(self):
        response = client.get("/search/text")
        assert response.status_code == 422


class TestImageSearch:
    @pytest.fixture
    def cat_jpeg_bytes(self):
        """Create a minimal test JPEG in memory."""
        img = Image.new("RGB", (64, 64), color=(120, 100, 80))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return buf.read()

    def test_returns_200_for_valid_image(self, cat_jpeg_bytes):
        response = client.post(
            "/search/image",
            files={"file": ("test.jpg", cat_jpeg_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_response_has_results_field(self, cat_jpeg_bytes):
        response = client.post(
            "/search/image",
            files={"file": ("test.jpg", cat_jpeg_bytes, "image/jpeg")},
        )
        data = response.json()
        assert "results" in data
        assert "count" in data

    def test_invalid_file_type_returns_400(self):
        response = client.post(
            "/search/image",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400
