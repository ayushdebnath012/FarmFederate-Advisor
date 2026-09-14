# FarmFederate Overleaf Package

Upload the contents of this folder to Overleaf and compile `main.tex`.

- Compiler: pdfLaTeX
- Bibliography: embedded in `main.tex`; no `.bib` file is required.
- Required figures are included under `plots/` and `disease_annotated/`.
- The three corpus photographs used by the architecture diagrams are in
  `plots/photos/`; the manuscript does not require a nested dataset bundle.
- `main.pdf` is the compiled current manuscript (8 pages, 20 references).
- Historical experiments in this directory are separate from paper compilation.

## Application and dataset additions

- `plots/app/` contains six original, unmodified prototype screenshots copied
  from the repository's `screenshots/` folder. `fig_app_phones.tex` lays them
  out in one row using the `\phoneshot` TikZ macro (defined in the `main.tex`
  preamble), which draws a phone outline and clips only the screen corners.
  The paper distinguishes demo values and simulated analytics/network views
  from measured LEAF results.
- `plots/dataset_examples/` contains one actual annotated crop per class from
  `Real Dataset/`, extracted with the training loader's 10% source-size padding.
  `provenance.json` records source filenames, SHA-256 hashes, annotation box
  indices, and crop bounds. No external or generated images were substituted.
- The dataset section documents the field source, supplied OBB format,
  preparation, generated notes, split policy, and unavailable collection metadata.
