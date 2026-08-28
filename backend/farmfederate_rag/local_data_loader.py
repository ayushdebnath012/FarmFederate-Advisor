#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
local_data_loader.py — Load local datasets into RAG Document objects.

Reads CSV files and image captions from the local `data/` folder and converts
them into Document objects compatible with the FarmFederate RAG pipeline.

Data sources handled:
  - Per-stress-type text.csv files (data/<stress_type>/text.csv)
  - Consolidated crop_stress_text_dataset.csv
  - captions.csv (agriculture-filtered)
  - Image captions linked to synthetic images
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from rag_core import Document, STRESS_TYPES

logger = logging.getLogger(__name__)

# Label index → stress type mapping (matches the Colab script)
LABEL_TO_STRESS = {i: st for i, st in enumerate(STRESS_TYPES)}

# Agriculture keywords for filtering captions.csv (which contains mixed content)
_AG_KEYWORDS = frozenset([
    "plant", "crop", "leaf", "leaves", "soil", "farm", "seed", "root",
    "flower", "fruit", "pest", "weed", "drought", "irrigat", "fertil",
    "harvest", "grain", "rice", "wheat", "maize", "cotton", "tomato",
    "potato", "bean", "stress", "disease", "blight", "wilt", "scorch",
    "chlorosis", "deficiency", "nutrient", "nitrogen", "phosphorus",
    "potassium", "insect", "fungal", "rust", "mildew", "rot",
])


def _content_hash(text: str) -> str:
    """Short hash for deduplication."""
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def _is_agriculture_text(text: str) -> bool:
    """Check if text is agriculture-related using keyword matching."""
    lower = text.lower()
    return any(kw in lower for kw in _AG_KEYWORDS)


class LocalDatasetLoader:
    """
    Loads local CSV datasets and converts rows into Document objects
    for the FarmFederate RAG pipeline's knowledge base.

    Usage:
        loader = LocalDatasetLoader(max_docs_per_class=200)
        docs = loader.load_all("data")
    """

    def __init__(self, max_docs_per_class: int = 200):
        """
        Args:
            max_docs_per_class: Maximum documents to keep per stress class
                                to prevent index imbalance. Set to -1 for no limit.
        """
        self.max_docs_per_class = max_docs_per_class

    def load_stress_text_csvs(self, data_dir: str) -> List[Document]:
        """
        Load per-stress-type text.csv files from subdirectories.

        Expects:  data_dir/<stress_type>/text.csv
        Columns:  text, labels, label_name, source

        Returns:
            List of Documents tagged with correct stress_type.
        """
        data_path = Path(data_dir)
        docs: List[Document] = []
        seen: Set[str] = set()

        for stress_type in STRESS_TYPES:
            csv_path = data_path / stress_type / "text.csv"
            if not csv_path.exists():
                logger.debug("No text.csv found for %s at %s", stress_type, csv_path)
                continue

            class_docs: List[Document] = []
            try:
                with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        text = (row.get("text") or "").strip()
                        if not text or len(text) < 20:
                            continue

                        h = _content_hash(text)
                        if h in seen:
                            continue
                        seen.add(h)

                        source = (row.get("source") or "local_text").strip()
                        class_docs.append(Document(
                            doc_id=f"local_{stress_type}_{h}",
                            content=text,
                            crop="general",
                            stress_type=stress_type,
                            region="global",
                            source=f"local_{source}",
                            metadata={"loader": "stress_text_csv"},
                        ))
            except Exception as e:
                logger.warning("Failed to read %s: %s", csv_path, e)
                continue

            # Apply per-class cap
            if self.max_docs_per_class > 0:
                class_docs = class_docs[: self.max_docs_per_class]

            docs.extend(class_docs)
            logger.info(
                "LocalDatasetLoader: loaded %d docs from %s",
                len(class_docs), csv_path,
            )

        return docs

    def load_crop_stress_dataset(self, csv_path: str) -> List[Document]:
        """
        Load the consolidated crop_stress_text_dataset.csv.

        Columns: text, label, label_name, source

        Returns:
            List of Documents with stress_type derived from label/label_name.
        """
        path = Path(csv_path)
        if not path.exists():
            logger.warning("crop_stress_text_dataset.csv not found at %s", path)
            return []

        docs: List[Document] = []
        seen: Set[str] = set()
        class_counts: Dict[str, int] = {st: 0 for st in STRESS_TYPES}

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = (row.get("text") or "").strip()
                    if not text or len(text) < 20:
                        continue

                    h = _content_hash(text)
                    if h in seen:
                        continue
                    seen.add(h)

                    # Determine stress type from label_name or label index
                    label_name = (row.get("label_name") or "").strip()
                    if label_name in STRESS_TYPES:
                        stress_type = label_name
                    else:
                        try:
                            label_idx = int(row.get("label", -1))
                            stress_type = LABEL_TO_STRESS.get(label_idx, "general")
                        except (ValueError, TypeError):
                            stress_type = "general"

                    # Apply per-class cap
                    if self.max_docs_per_class > 0:
                        if class_counts.get(stress_type, 0) >= self.max_docs_per_class:
                            continue
                        class_counts[stress_type] = class_counts.get(stress_type, 0) + 1

                    source = (row.get("source") or "local_dataset").strip()
                    docs.append(Document(
                        doc_id=f"local_csd_{h}",
                        content=text,
                        crop="general",
                        stress_type=stress_type,
                        region="global",
                        source=f"local_{source}",
                        metadata={"loader": "crop_stress_dataset"},
                    ))
        except Exception as e:
            logger.warning("Failed to read %s: %s", path, e)

        logger.info(
            "LocalDatasetLoader: loaded %d docs from crop_stress_text_dataset.csv",
            len(docs),
        )
        return docs

    def load_captions(self, csv_path: str) -> List[Document]:
        """
        Load captions.csv, filtering for agriculture-relevant entries.

        Columns: text (or caption), label/label_name (optional)

        Returns:
            List of agriculture-filtered Document objects.
        """
        path = Path(csv_path)
        if not path.exists():
            logger.warning("captions.csv not found at %s", path)
            return []

        docs: List[Document] = []
        seen: Set[str] = set()

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try 'text' column first, then 'caption'
                    text = (row.get("text") or row.get("caption") or "").strip()
                    if not text or len(text) < 20:
                        continue

                    # Filter for agriculture-related content
                    if not _is_agriculture_text(text):
                        continue

                    h = _content_hash(text)
                    if h in seen:
                        continue
                    seen.add(h)

                    # Try to determine stress type if available
                    label_name = (row.get("label_name") or "").strip()
                    if label_name in STRESS_TYPES:
                        stress_type = label_name
                    else:
                        try:
                            label_idx = int(row.get("label", -1))
                            stress_type = LABEL_TO_STRESS.get(label_idx, "general")
                        except (ValueError, TypeError):
                            stress_type = "general"

                    docs.append(Document(
                        doc_id=f"local_cap_{h}",
                        content=text,
                        crop="general",
                        stress_type=stress_type,
                        region="global",
                        source="local_caption",
                        metadata={"loader": "captions_csv"},
                    ))
        except Exception as e:
            logger.warning("Failed to read %s: %s", path, e)

        logger.info(
            "LocalDatasetLoader: loaded %d agriculture-relevant captions", len(docs),
        )
        return docs

    def load_all(self, data_dir: str) -> List[Document]:
        """
        Load all local datasets and return combined, deduplicated documents.

        Args:
            data_dir: Path to the data directory (e.g., "data" or absolute path).

        Returns:
            Combined list of Document objects from all local sources.
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.warning("Data directory not found: %s", data_path)
            return []

        all_docs: List[Document] = []
        seen_hashes: Set[str] = set()

        # 1. Per-stress-type text CSVs
        stress_docs = self.load_stress_text_csvs(data_dir)
        for doc in stress_docs:
            h = _content_hash(doc.content)
            if h not in seen_hashes:
                seen_hashes.add(h)
                all_docs.append(doc)

        # 2. Consolidated crop_stress_text_dataset.csv
        csd_path = data_path / "crop_stress_text_dataset.csv"
        if csd_path.exists():
            csd_docs = self.load_crop_stress_dataset(str(csd_path))
            for doc in csd_docs:
                h = _content_hash(doc.content)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    all_docs.append(doc)

        # 3. Captions CSV
        cap_path = data_path / "captions.csv"
        if cap_path.exists():
            cap_docs = self.load_captions(str(cap_path))
            for doc in cap_docs:
                h = _content_hash(doc.content)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    all_docs.append(doc)

        logger.info(
            "LocalDatasetLoader.load_all: %d unique documents from %s",
            len(all_docs), data_path,
        )
        return all_docs


class ImageCaptionLoader:
    """
    Creates Document objects from image captions, storing image paths
    in metadata for reference. The text content (caption) is used for
    retrieval in the RAG pipeline.

    Usage:
        loader = ImageCaptionLoader()
        docs = loader.load_image_captions("data")
    """

    def load_image_captions(
        self, data_dir: str, max_per_class: int = 50
    ) -> List[Document]:
        """
        Scan image subdirectories and create Documents with captions.

        For each stress type, reads the text.csv to get captions and
        cross-references with available images in the images/ subdirectory.

        Args:
            data_dir:      Path to data/ directory.
            max_per_class: Maximum image-caption docs per stress class.

        Returns:
            List of Documents with image_path in metadata.
        """
        data_path = Path(data_dir)
        docs: List[Document] = []

        for stress_type in STRESS_TYPES:
            img_dir = data_path / stress_type / "images"
            csv_path = data_path / stress_type / "text.csv"

            if not img_dir.exists() or not csv_path.exists():
                continue

            # Get available image filenames
            image_files = sorted([
                f.name for f in img_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")
            ])

            if not image_files:
                continue

            # Read captions from text.csv (use blip_caption source entries)
            captions: List[str] = []
            try:
                with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        source = (row.get("source") or "").strip()
                        if source == "blip_caption":
                            text = (row.get("text") or "").strip()
                            if text and len(text) >= 10:
                                captions.append(text)
            except Exception as e:
                logger.warning("Failed to read captions from %s: %s", csv_path, e)
                continue

            # Pair images with captions (round-robin if counts differ)
            count = min(max_per_class, len(image_files), max(1, len(captions)))
            for i in range(count):
                caption = captions[i % len(captions)] if captions else f"{stress_type} crop stress image"
                img_name = image_files[i % len(image_files)]
                img_path = str(img_dir / img_name)

                docs.append(Document(
                    doc_id=f"local_img_{stress_type}_{i:04d}",
                    content=caption,
                    crop="general",
                    stress_type=stress_type,
                    region="global",
                    source="local_image_caption",
                    metadata={
                        "loader": "image_caption",
                        "image_path": img_path,
                        "image_name": img_name,
                    },
                ))

            logger.info(
                "ImageCaptionLoader: %d image-caption docs for %s",
                count, stress_type,
            )

        logger.info(
            "ImageCaptionLoader: %d total image-caption documents", len(docs),
        )
        return docs
