"""
Tests for local dataset integration with the FarmFederate RAG pipeline.
"""

import os
import sys
import csv
import tempfile
import shutil
from pathlib import Path

import pytest

# Ensure backend/farmfederate_rag is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "farmfederate_rag"))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from rag_core import Document, STRESS_TYPES
from local_data_loader import (
    LocalDatasetLoader,
    ImageCaptionLoader,
    _content_hash,
    _is_agriculture_text,
)


# ---------------------------------------------------------------------------
# Fixtures: create a temporary data directory with sample CSV files
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a temporary data directory mimicking the real structure."""
    # Create per-stress-type subdirectories with text.csv and images/
    for i, stress_type in enumerate(STRESS_TYPES):
        stress_dir = tmp_path / stress_type
        stress_dir.mkdir()

        # text.csv
        csv_path = stress_dir / "text.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "labels", "label_name", "source"])
            writer.writeheader()
            for j in range(10):
                writer.writerow({
                    "text": f"Sample {stress_type} observation number {j} with detailed description of symptoms",
                    "labels": f"[{i}]",
                    "label_name": stress_type,
                    "source": "blip_caption" if j % 2 == 0 else "pvvqa_argilla",
                })
            # Add a short text that should be filtered out
            writer.writerow({
                "text": "too short",
                "labels": f"[{i}]",
                "label_name": stress_type,
                "source": "blip_caption",
            })

        # images/ with dummy files
        img_dir = stress_dir / "images"
        img_dir.mkdir()
        for j in range(3):
            (img_dir / f"syn_{j:05d}.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 100)

    # Create crop_stress_text_dataset.csv
    csd_path = tmp_path / "crop_stress_text_dataset.csv"
    with open(csd_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "label_name", "source"])
        writer.writeheader()
        for i, stress_type in enumerate(STRESS_TYPES):
            for j in range(5):
                writer.writerow({
                    "text": f"Consolidated dataset entry for {stress_type} class index {i} item {j} with extra description",
                    "label": str(i),
                    "label_name": stress_type,
                    "source": "blip_caption",
                })

    # Create captions.csv with mixed content
    cap_path = tmp_path / "captions.csv"
    with open(cap_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "label_name"])
        writer.writeheader()
        writer.writerow({
            "text": "A wilting crop plant with drooping leaves showing brown spots on a field",
            "label": "0",
            "label_name": "water_stress",
        })
        writer.writerow({
            "text": "A landscape photograph of mountains and rivers with no agriculture context",
            "label": "-1",
            "label_name": "",
        })
        writer.writerow({
            "text": "Nitrogen deficiency observed in wheat leaves yellowing from base upward",
            "label": "1",
            "label_name": "nutrient_def",
        })

    # Create dataset_info.json
    (tmp_path / "dataset_info.json").write_text(
        '{"sources": ["test"], "counts": {}, "total": 0}'
    )

    return tmp_path


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_content_hash_consistent(self):
        assert _content_hash("hello") == _content_hash("hello")

    def test_content_hash_different(self):
        assert _content_hash("hello") != _content_hash("world")

    def test_is_agriculture_text_positive(self):
        assert _is_agriculture_text("The crop plant shows drought stress symptoms")

    def test_is_agriculture_text_negative(self):
        assert not _is_agriculture_text("A beautiful sunset over the ocean")


class TestLocalDatasetLoader:
    def test_load_stress_text_csvs(self, sample_data_dir):
        loader = LocalDatasetLoader(max_docs_per_class=-1)
        docs = loader.load_stress_text_csvs(str(sample_data_dir))

        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)
        # Should have docs from all 5 stress types
        stress_types_found = {d.stress_type for d in docs}
        assert stress_types_found == set(STRESS_TYPES)

    def test_load_stress_text_csvs_filters_short(self, sample_data_dir):
        loader = LocalDatasetLoader(max_docs_per_class=-1)
        docs = loader.load_stress_text_csvs(str(sample_data_dir))

        # "too short" entries should be filtered
        for d in docs:
            assert len(d.content) >= 20

    def test_load_stress_text_csvs_deduplicates(self, sample_data_dir):
        loader = LocalDatasetLoader(max_docs_per_class=-1)
        docs = loader.load_stress_text_csvs(str(sample_data_dir))
        contents = [d.content for d in docs]
        assert len(contents) == len(set(contents))

    def test_max_docs_per_class_limit(self, sample_data_dir):
        loader = LocalDatasetLoader(max_docs_per_class=3)
        docs = loader.load_stress_text_csvs(str(sample_data_dir))

        # Each class should have at most 3 docs
        from collections import Counter
        counts = Counter(d.stress_type for d in docs)
        for st, count in counts.items():
            assert count <= 3, f"{st} has {count} docs, expected <= 3"

    def test_load_crop_stress_dataset(self, sample_data_dir):
        loader = LocalDatasetLoader(max_docs_per_class=-1)
        csv_path = str(sample_data_dir / "crop_stress_text_dataset.csv")
        docs = loader.load_crop_stress_dataset(csv_path)

        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)
        # All docs should have valid stress types
        for d in docs:
            assert d.stress_type in STRESS_TYPES or d.stress_type == "general"

    def test_load_crop_stress_dataset_missing_file(self):
        loader = LocalDatasetLoader()
        docs = loader.load_crop_stress_dataset("/nonexistent/path.csv")
        assert docs == []

    def test_load_captions_filters_agriculture(self, sample_data_dir):
        loader = LocalDatasetLoader()
        cap_path = str(sample_data_dir / "captions.csv")
        docs = loader.load_captions(cap_path)

        # Should include agriculture-related texts only
        assert len(docs) >= 1
        for d in docs:
            assert any(
                kw in d.content.lower()
                for kw in ["crop", "plant", "leaf", "drought", "nitrogen", "wheat"]
            )

    def test_load_all_returns_documents(self, sample_data_dir):
        loader = LocalDatasetLoader(max_docs_per_class=-1)
        docs = loader.load_all(str(sample_data_dir))

        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)
        # Should have doc_id, content, stress_type, source
        for d in docs:
            assert d.doc_id
            assert d.content
            assert d.source.startswith("local_")

    def test_load_all_deduplicates_across_sources(self, sample_data_dir):
        loader = LocalDatasetLoader(max_docs_per_class=-1)
        docs = loader.load_all(str(sample_data_dir))
        contents = [d.content for d in docs]
        assert len(contents) == len(set(contents))

    def test_load_all_nonexistent_dir(self):
        loader = LocalDatasetLoader()
        docs = loader.load_all("/nonexistent/path")
        assert docs == []


class TestImageCaptionLoader:
    def test_load_image_captions(self, sample_data_dir):
        loader = ImageCaptionLoader()
        docs = loader.load_image_captions(str(sample_data_dir), max_per_class=2)

        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)
        for d in docs:
            assert d.source == "local_image_caption"
            assert "image_path" in d.metadata
            assert d.stress_type in STRESS_TYPES

    def test_load_image_captions_max_per_class(self, sample_data_dir):
        loader = ImageCaptionLoader()
        docs = loader.load_image_captions(str(sample_data_dir), max_per_class=1)

        from collections import Counter
        counts = Counter(d.stress_type for d in docs)
        for st, count in counts.items():
            assert count <= 1


class TestDocumentFields:
    """Verify Document objects have correct metadata."""

    def test_document_fields_stress_csv(self, sample_data_dir):
        loader = LocalDatasetLoader(max_docs_per_class=1)
        docs = loader.load_stress_text_csvs(str(sample_data_dir))
        if docs:
            d = docs[0]
            assert d.crop == "general"
            assert d.region == "global"
            assert d.stress_type in STRESS_TYPES
            assert "loader" in d.metadata
            assert d.metadata["loader"] == "stress_text_csv"

    def test_document_fields_crop_stress_dataset(self, sample_data_dir):
        loader = LocalDatasetLoader(max_docs_per_class=1)
        docs = loader.load_crop_stress_dataset(
            str(sample_data_dir / "crop_stress_text_dataset.csv")
        )
        if docs:
            d = docs[0]
            assert d.metadata["loader"] == "crop_stress_dataset"
            assert d.doc_id.startswith("local_csd_")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
