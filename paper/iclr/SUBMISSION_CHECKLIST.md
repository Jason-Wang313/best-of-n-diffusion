# ICLR Initial Submission Checklist

Target: ICLR 2027 initial submission package. As of 2026-06-09, no official ICLR 2027 author guide was found, so this package uses the official ICLR 2026 author guide as the formatting proxy: anonymous double-blind submission, official LaTeX style, and a 9-page main-text cap before references.

## Build

- Source: `paper/iclr/main.tex`, `paper/iclr/appendix.tex`, `paper/iclr/references.bib`.
- Style proxy: official `iclr2026_conference.sty` and `iclr2026_conference.bst`.
- Figures: copied from audited `results/figures/*.png` into `paper/iclr/figures/`.
- Command: `python scripts/build_iclr_paper.py`.
- Output: `paper/iclr/main.pdf`.
- Page-count gate: the build script reads the `maintextend` LaTeX label and fails if the page before references exceeds 9.
- Observed build on 2026-06-09: main-text label page 5; total PDF pages 9.
- On this machine, MiKTeX's `latexmk` shim lacks Perl, so the wrapper automatically falls back to `pdflatex`/`bibtex`.

## Anonymity

- `\iclrfinalcopy` remains commented.
- Main PDF should show anonymous review authors only.
- No acknowledgments are included.
- No personal GitHub links, author names, local Windows paths, or institutional identifiers should appear in `main.tex`, `appendix.tex`, or the generated PDF.
- Code is described as anonymous supplementary material.

## Claims

- No new experiments are introduced.
- Main text only promotes claims supported by the full-run audit artifacts.
- Universal high-`N` improvement, real-robot validation, hardware safety certification, and full visual-policy validation are explicitly excluded.
- Strong wording is gated on true-DDPM and PushT rollout-metric evidence.

## Supplement

- Appendix is included after references in `main.pdf`.
- `paper/iclr/supplement/README.md` describes an anonymous supplementary package.
- Supplement content should contain source code, `results/`, `docs/`, `tests/`, and reproduction commands, but no author-identifying links or local absolute paths.

## Final Commands

```bash
python scripts/build_iclr_paper.py
bash scripts/run_claim_audit.sh
python -m pytest -q
rg -n "F(?:INAL RERUN)|T(?:ODO)|placehold(?:er)|almost\s+100|100[%]|validated on real robot[s]|trajectory search always help[s]" paper docs README.md
```

## Manual Checks Before Upload

- Open `paper/iclr/main.pdf` and verify figures are legible.
- Confirm the main-text page count before references is at most 9.
- Confirm references and appendix begin after the main text.
- Confirm every main numerical claim has a table, figure, or appendix pointer.
- Confirm the anonymous supplement contains tracked source artifacts only.
