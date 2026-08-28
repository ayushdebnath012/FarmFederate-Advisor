#!/usr/bin/env python3
"""Justify the template-composed note corpus with real-practice citations.

The paper states the notes are templated from phrase pools and treats results on
them as upper bounds. What it does not say is why a templated corpus is a
defensible proxy at all. Field plant-health observations are conventionally
recorded against controlled vocabularies and fixed severity scales rather than
as free prose, which is documented and citable.

The distinction matters and is kept explicit: the literature supports the
*structure* of real records being controlled, not their surface wording being
drawn from a closed pool. Our corpus is far more closed than any real one --
3000 rows collapse to 131 distinct sentences -- so the upper-bound framing
stays.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = chr(92)
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]
END = B + "end{thebibliography}"

ITEMS = [
    ("Shrestha2010CropOntology",
     "R.~Shrestha, E.~Arnaud, R.~Mauleon, M.~Senger, G.~F. Davenport, D.~Hancock\n"
     "\\emph{et al.}, ``Multifunctional crop trait ontology for breeders' data: field\n"
     "book, annotation, data discovery and semantic enrichment of the literature,''\n"
     "\\emph{AoB PLANTS}, vol.~2010, art.~plq008, 2010."),
    ("Andres2021FAIRVocab",
     "L.~Andr\\'es-Hern\\'andez, R.~Azman~Halimi, R.~Mauleon, S.~Mayes, A.~Baten, and\n"
     "G.~J. King, ``Challenges for {FAIR}-compliant description and comparison of crop\n"
     "phenotype data with standardized controlled vocabularies,'' \\emph{Database},\n"
     "vol.~2021, art.~baab028, 2021."),
    ("Bock2021Severity",
     "C.~H. Bock, K.-S. Chiang, and E.~M. Del~Ponte, ``Plant disease severity estimated\n"
     "visually: a century of research, best practices, and opportunities for improving\n"
     "methods and practices to maximize accuracy,'' \\emph{Trop.\\ Plant Pathol.},\n"
     "vol.~47, no.~1, pp.~25--42, 2021."),
    ("DelPonte2017SAD",
     "E.~M. Del~Ponte, S.~J. Pethybridge, C.~H. Bock, S.~J. Michereff, F.~J. Machado,\n"
     "and P.~Spolti, ``Standard area diagrams for aiding severity estimation:\n"
     "scientometrics, pathosystems, and methodological trends,''\n"
     "\\emph{Phytopathology}, vol.~107, no.~10, pp.~1161--1174, 2017."),
]

OLD = ("Notes are\n"
       "templated from manually written phrase pools, not independent farmer prose.")
NEW = ("Notes are\n"
       "templated from manually written phrase pools, not independent farmer prose.\n"
       "Recording field observations against a controlled vocabulary rather than as\n"
       "free text is itself standard practice --- crop trait ontologies supply\n"
       "harmonized trait/method/scale terms for breeders' field\n"
       "books~\\cite{Shrestha2010CropOntology,Andres2021FAIRVocab}, and disease\n"
       "severity is conventionally scored on fixed ordinal scales and standard area\n"
       "diagrams~\\cite{Bock2021Severity,DelPonte2017SAD} --- so a template-composed\n"
       "corpus is a reasonable proxy for the \\emph{structure} of real records. It is\n"
       "not a proxy for their diversity: 3{,}000 annotation rows here collapse to 131\n"
       "distinct sentences, 43 of them shared across classes, so every note-side\n"
       "number in this paper upper-bounds what free-form field notes would give.")


def main() -> None:
    block = "\n".join(B + "bibitem{" + k + "}\n" + v + "\n" for k, v in ITEMS)
    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        notes = []
        if ITEMS[0][0] in s:
            notes.append("bibitems already present")
        else:
            assert s.count(END) == 1, f"{tex}: {s.count(END)} bibliography ends"
            s = s.replace(END, block + END)
            notes.append(f"{len(ITEMS)} bibitems added")
        if "controlled vocabulary rather than as" in s:
            notes.append("sentence already present")
        elif s.count(OLD) == 1:
            s = s.replace(OLD, NEW)
            notes.append("justification added")
        else:
            notes.append(f"ANCHORx{s.count(OLD)}")
        p.write_text(s, encoding="utf-8")
        print(f"  {tex}: {'; '.join(notes)}")


if __name__ == "__main__":
    main()
