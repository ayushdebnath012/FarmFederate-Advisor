#!/usr/bin/env python3
"""Remove the legacy five-class vocabulary from the live code.

The only disease classes are the ones in data_final/label_schema.json:
LEAF_BLIGHT, LEAF_HOPPERS, LEAF_RUST, LOOPER_CATERPILLARS, MOSQUITO_BUG.

The old set (gray_blight, helopeltis, algal_leaf_spot, brown_blight,
red_leaf_spot) came from a 2026-04-14 override that disagreed with the schema.
This removes it from the code paths that are actually executed:

  tea_train.py     header docstring; _KW symptom vocabulary, which was keyed to
                   the old semantics and feeds generate_text_data()
  base module      ORIGINAL_TEA_CLASSES (defined, never referenced)

DISEASE_TO_STRESS is deliberately left alone: it maps by pathology and
contradicts the id ordering, so it is the record that class 1 and class 3 are
contested. Deleting it would erase the evidence of that disagreement.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEA = ROOT / "tea_train.py"
BASE = ROOT / "backend" / "FarmFederate_Colab_Complete.py"

OLD_DOC = "Classes : gray_blight | helopeltis | algal_leaf_spot | brown_blight | red_leaf_spot"
NEW_DOC = ("Classes : LEAF_BLIGHT | LEAF_HOPPERS | LEAF_RUST | LOOPER_CATERPILLARS "
           "| MOSQUITO_BUG\n          (data_final/label_schema.json, "
           "farmfederate_stress_v2_direct_yolo_ids)")

NEW_KW = '''_KW = {
    0: {  # LEAF_BLIGHT
        "obs": ["greyish-brown blotches developing on mature leaves",
                "necrosis spreading inward from the leaf margin",
                "affected leaves drying from the tip backwards"],
        "sym": ["irregular brown lesions with darker margins",
                "concentric zonation within the older lesions",
                "lesions coalescing into large dry blighted areas"],
        "cond": ["high relative humidity sustained over several days",
                 "dense canopy reducing light penetration to lower leaves",
                 "rain splash dispersing spores between bushes"],
        "ind": ["lesion area exceeding a third of the leaf lamina",
                "several leaves on the shoot showing the same pattern",
                "incidence spreading across multiple rows in the block"],
    },
    1: {  # LEAF_HOPPERS
        "obs": ["marginal yellowing progressing to a dry brown rim",
                "scorched leaf edges that curl upward when dry",
                "stunted flush with shortened internodes"],
        "sym": ["small wedge-shaped insects disturbed from the underside",
                "fine stippling visible against the light",
                "tip and margin burn without a defined lesion border"],
        "cond": ["warm dry spells raising hopper activity",
                 "tender flush growth attracting feeding",
                 "sheltered blocks with little air movement"],
        "ind": ["damage on more than a tenth of the new shoots",
                "plucking table visibly checked in growth",
                "damage concentrated on the younger flush"],
    },
    2: {  # LEAF_RUST
        "obs": ["raised circular spots on the upper leaf surface",
                "spots most prominent in bright light",
                "scattered spots of varying size across the lamina"],
        "sym": ["orange to rust-red velvety growth on the affected patches",
                "raised circular spots slightly rough to the touch",
                "rust-coloured patches with a felt-like texture"],
        "cond": ["high humidity with abundant light",
                 "cool misty weather common at higher elevation",
                 "persistent leaf wetness from dew"],
        "ind": ["multiple spots per leaf reducing photosynthetic area",
                "spread moving along the stem to the petiole",
                "older leaves more affected than the new growth"],
    },
    3: {  # LOOPER_CATERPILLARS
        "obs": ["chewed leaf margins with irregular notches",
                "partial defoliation of the shoot with midribs left intact",
                "feeding damage visible from a distance along the row"],
        "sym": ["irregular holes through the lamina between the veins",
                "dark frass pellets caught in the leaf axils",
                "leaf tissue removed leaving a skeletonised outline"],
        "cond": ["warm weather following the monsoon",
                 "the section borders an older unpruned block",
                 "flush growth providing abundant soft foliage"],
        "ind": ["defoliation visible on several bushes in the block",
                "fresh feeding damage on the current flush",
                "larvae found on the underside during inspection"],
    },
    4: {  # MOSQUITO_BUG
        "obs": ["sunken dark feeding punctures on the young shoot",
                "shoot tip wilting above the damaged zone",
                "damage concentrated on tender growth"],
        "sym": ["corky brown scars where earlier punctures have healed",
                "pinhole-size marks enlarging into angular sunken lesions",
                "blackening of the terminal bud after feeding"],
        "cond": ["warm humid weather with intermittent rain",
                 "adjacent shade trees providing refuge",
                 "tender flush available through the plucking round"],
        "ind": ["fresh feeding marks on the current flush",
                "secondary infection entering through the wounds",
                "shoot dieback reducing the pluckable surface"],
    },
}'''


def replace_block(text: str, start_marker: str, end_marker: str, new: str) -> str:
    a = text.index(start_marker)
    b = text.index(end_marker, a)
    return text[:a] + new + "\n\n" + text[b:]


def main() -> None:
    s = TEA.read_text(encoding="utf-8")
    notes = []
    if OLD_DOC in s:
        s = s.replace(OLD_DOC, NEW_DOC)
        notes.append("docstring")
    if 'gray_blight (Pestalotiopsis' in s:
        s = replace_block(s, "_KW = {", "_TEMPLATES = [", NEW_KW)
        notes.append("_KW vocabulary")
    TEA.write_text(s, encoding="utf-8")
    print(f"  tea_train.py: {', '.join(notes) or 'already clean'}")

    b = BASE.read_text(encoding="utf-8")
    old_orig = ("ORIGINAL_TEA_CLASSES = [\n"
                "    'gray_blight', 'helopeltis', 'algal_leaf_spot', "
                "'brown_blight', 'red_leaf_spot'\n]")
    if old_orig in b:
        b = b.replace(old_orig + "\n", "")
        BASE.write_text(b, encoding="utf-8")
        print("  base module: ORIGINAL_TEA_CLASSES removed")
    else:
        print("  base module: ORIGINAL_TEA_CLASSES not found verbatim")


if __name__ == "__main__":
    main()
