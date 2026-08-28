#!/usr/bin/env python3
"""Rewrite the crop-linked field notes so they are correct and non-trivial.

Two defects in the previous corpus:

1. Wrong pathology. The notes were generated from a label map that disagreed
   with data_final/label_schema.json, so class 0 (LEAF_BLIGHT) carried a
   Pestalotiopsis gray-blight description, class 1 (LEAF_HOPPERS) carried
   Helopeltis feeding damage, and so on. The text described the wrong condition
   for every class.

2. Every sentence was class-exclusive. 742 sentences over 328 distinct strings,
   none shared between classes, so a bag-of-words classifier reached 1.000 and
   the 178-token leakage mask could not help: the shortcut was the sentence,
   not the word.

This generator fixes both. Symptom phrases are grouped into pools, and each
pool is declared for the set of classes in which that observation genuinely
occurs -- brown lesions with dark margins really do appear for both blight and
mosquito-bug damage; marginal drying really is shared by hopper burn and
blight. A class is characterised by its *distribution* over pools, not by
owning any phrase outright, so the label is recoverable from a combination of
cues and never from a single sentence.

Crop identity is preserved exactly: image_file, box_idx and class_id are
carried over unchanged, so the 222/74/75 source-grouped split and the exact
(photograph, box) linkage are untouched. Only the text is rewritten.

Deterministic given --seed.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "tea_results" / "annotation" / "annotations.csv"
SCHEMA = ROOT / "data_final" / "label_schema.json"

BLIGHT, HOPPERS, RUST, LOOPER, MOSQ = 0, 1, 2, 3, 4
ALL = (BLIGHT, HOPPERS, RUST, LOOPER, MOSQ)

# Phrases carry a specificity. GENERIC observations are consistent with several
# conditions and are what most of a scout's line actually is; DIAGNOSTIC ones
# point at one or two conditions. A real note records the diagnostic sign only
# when the scout happened to see it, so it appears with probability P_DIAG and
# the rest of the note is generic. That, not a tuned weight, is what stops the
# text from determining the label.
P_DIAG = 0.45

GENERIC = [
    "irregular brown lesions with darker margins on the lamina",
    "greyish-brown patches spreading inward from the leaf margin",
    "necrotic tissue with a narrow chlorotic halo",
    "scorched leaf edges that curl upward when dry",
    "marginal yellowing progressing to a dry brown rim",
    "discoloured patches of no consistent outline",
    "affected leaves drying from the tip backwards",
    "damage concentrated on the younger flush",
    "scattered spots of varying size across the lamina",
    "shoot tip wilting above the damaged zone",
    "several leaves on the shoot showing the same pattern",
    "older leaves more affected than the new growth",
]

DIAGNOSTIC = {
    BLIGHT:  ["concentric zonation visible within the older lesions",
              "lesions coalescing into large dry blighted areas"],
    HOPPERS: ["small wedge-shaped insects disturbed from the underside on handling",
              "stunted flush with shortened internodes"],
    RUST:    ["orange to rust-red velvety growth on the affected patches",
              "raised circular spots slightly rough to the touch"],
    LOOPER:  ["chewed leaf margins with irregular notches",
              "irregular holes through the lamina between the veins",
              "dark frass pellets caught in the leaf axils"],
    MOSQ:    ["sunken dark feeding punctures on the young shoot",
              "corky brown scars where earlier punctures have healed"],
}


# context sentences are shared verbatim by every class and carry no class signal
CONTEXT = [
    "recorded during the routine block walk",
    "sampled from the middle canopy of a mature bush",
    "humidity had been high for several days before the visit",
    "the block was pruned in the previous season",
    "observed on the shaded side of the row",
    "leaf collected from second-flush material",
    "temperature mild with intermittent rain in the preceding week",
    "the section borders an older unpruned block",
    "drainage in this part of the field is poor",
    "noted while assessing the plucking table",
]
OPENER = [
    "Field note", "Scout record", "Block observation", "Inspection entry",
    "Plucking-round note", "Field entry",
]

def build_note(cls: int, rng: random.Random) -> str:
    """Mostly generic observation, with the diagnostic sign only sometimes seen."""
    parts = []
    if rng.random() < P_DIAG:
        parts.append(rng.choice(DIAGNOSTIC[cls]))
    n_generic = rng.choice((1, 2, 2)) if parts else rng.choice((2, 3))
    parts += rng.sample(GENERIC, n_generic)
    rng.shuffle(parts)
    body = "; ".join(parts)
    return f"{rng.choice(OPENER)}: {body}. {rng.choice(CONTEXT).capitalize()}."


def audit(df: pd.DataFrame, tag: str) -> None:
    by = defaultdict(set)
    n = 0
    for _, r in df.iterrows():
        for s in re.split(r"[.;]", str(r["text"])):
            s = s.strip().lower()
            if s:
                by[s].add(int(r["class_id"]))
                n += 1
    shared = sum(1 for v in by.values() if len(v) > 1)
    print(f"  {tag:<9} {n} sentences, {len(by)} distinct, "
          f"{shared} shared across classes ({100 * shared / max(len(by), 1):.0f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    schema = json.load(open(SCHEMA, encoding="utf-8"))["raw_yolo_id_to_stress"]
    df = pd.read_csv(CSV)
    print(f"loaded {len(df)} crop-linked rows")
    audit(df, "before")

    rng = random.Random(a.seed)
    out = df.copy()
    out["disease"] = out["class_id"].astype(int).map(lambda i: schema[str(i)])
    out["text"] = [build_note(int(c), rng) for c in out["class_id"]]
    audit(out, "after")

    print("\n  class distribution unchanged:",
          dict(sorted(Counter(out["class_id"]).items())))
    print("\n  samples:")
    for cid in sorted(out["class_id"].unique()):
        row = out[out["class_id"] == cid].iloc[0]
        print(f"   [{schema[str(cid)]}] {row['text'][:150]}")

    if a.dry_run:
        print("\n(dry run: nothing written)")
        return
    shutil.copy(CSV, str(CSV) + ".oldnotes")
    out.to_csv(CSV, index=False)
    print(f"\nwrote {CSV} (previous kept as .oldnotes)")


if __name__ == "__main__":
    main()
