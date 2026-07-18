"""
tests/test_embeddings.py
─────────────────────────
Unit tests for the CLIP embedding module.

Run with:
  source .venv/bin/activate
  pytest tests/test_embeddings.py -v

Learning notes:
  - pytest discovers tests automatically: any function starting with test_ in
    files named test_*.py or *_test.py is a test.
  - We test PROPERTIES of the output (shape, normalization, similarity ordering)
    not exact values — ML model outputs are not perfectly deterministic.
  - The cross-modal test is the most important: it verifies CLIP's core promise
    that related text and images have higher similarity than unrelated ones.
"""

import io

import pytest
import requests
import torch
from PIL import Image

from app.embeddings import cosine_similarity, encode_image, encode_text

VECTOR_DIM = 512


class TestEncodeText:
    def test_returns_list_of_floats(self):
        result = encode_text("a cat")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_correct_dimension(self):
        result = encode_text("hello world")
        assert len(result) == VECTOR_DIM

    def test_is_normalized(self):
        """L2 norm of the vector should be very close to 1.0."""
        result = encode_text("a sunset over the ocean")
        norm = sum(x**2 for x in result) ** 0.5
        assert abs(norm - 1.0) < 1e-4, f"Vector not normalized: norm={norm}"

    def test_different_texts_produce_different_vectors(self):
        vec1 = encode_text("a cat")
        vec2 = encode_text("a rocket ship")
        # Cosine similarity should be less than 0.99 (not identical)
        sim = cosine_similarity(vec1, vec2)
        assert sim < 0.99

    def test_similar_texts_are_close(self):
        vec1 = encode_text("a dog")
        vec2 = encode_text("a puppy")
        vec3 = encode_text("a spaceship on Mars")
        sim_close = cosine_similarity(vec1, vec2)
        sim_far = cosine_similarity(vec1, vec3)
        assert sim_close > sim_far, "Related texts should be more similar"


class TestEncodeImage:
    @pytest.fixture
    def cat_image(self):
        """Download a real cat image for testing."""
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/160px-Cat_November_2010-1a.jpg"
        resp = requests.get(url, timeout=10)
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def test_returns_list_of_floats(self, cat_image):
        result = encode_image(cat_image)
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_correct_dimension(self, cat_image):
        result = encode_image(cat_image)
        assert len(result) == VECTOR_DIM

    def test_is_normalized(self, cat_image):
        result = encode_image(cat_image)
        norm = sum(x**2 for x in result) ** 0.5
        assert abs(norm - 1.0) < 1e-4


class TestCrossModalAlignment:
    """
    The most important tests — verifying CLIP's core multimodal promise:
    matching text-image pairs should have HIGHER similarity than non-matching pairs.
    """

    @pytest.fixture(scope="class")
    def cat_image(self):
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/160px-Cat_November_2010-1a.jpg"
        resp = requests.get(url, timeout=10)
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def test_cat_text_closer_to_cat_image_than_dog_text(self, cat_image):
        img_vec = encode_image(cat_image)
        cat_text_vec = encode_text("a cat")
        dog_text_vec = encode_text("a dog")

        cat_sim = cosine_similarity(img_vec, cat_text_vec)
        dog_sim = cosine_similarity(img_vec, dog_text_vec)

        assert cat_sim > dog_sim, (
            f"CLIP cross-modal alignment failed: "
            f"cat_sim={cat_sim:.4f} should be > dog_sim={dog_sim:.4f}"
        )
