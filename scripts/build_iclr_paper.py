#!/usr/bin/env python
"""Build the anonymous ICLR paper and enforce the main-text page limit."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


MAIN_PAGE_LIMIT = 9
DEFAULT_DESKTOP_PDF = Path.home() / "OneDrive" / "Desktop" / "best of n diffusion policy-v2.pdf"


def run(cmd: list[str], cwd: Path) -> None:
    print(f"+ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def parse_label_page(aux_path: Path, label: str) -> int:
    text = aux_path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(rf"\\newlabel\{{{re.escape(label)}\}}\{{.*?\}}\{{(\d+)\}}", re.DOTALL)
    match = pattern.search(text)
    if not match:
        raise RuntimeError(f"Could not find LaTeX label {label!r} in {aux_path}")
    return int(match.group(1))


def total_pdf_pages(pdf_path: Path) -> int | None:
    pdfinfo = shutil.which("pdfinfo")
    if not pdfinfo:
        return None
    proc = subprocess.run(
        [pdfinfo, str(pdf_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    return None


def usable_latexmk() -> str | None:
    latexmk = shutil.which("latexmk")
    if not latexmk:
        return None
    proc = subprocess.run([latexmk, "-version"], capture_output=True, text=True)
    if proc.returncode != 0:
        print("latexmk was found but is not usable in this environment; falling back to pdflatex/bibtex.")
        return None
    return latexmk


def build_with_latexmk(paper_dir: Path) -> None:
    latexmk = usable_latexmk()
    if not latexmk:
        raise FileNotFoundError("latexmk")
    run([latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=paper_dir)


def build_with_pdflatex(paper_dir: Path) -> None:
    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    if not pdflatex or not bibtex:
        raise FileNotFoundError("pdflatex and bibtex are required when latexmk is unavailable")
    run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=paper_dir)
    run([bibtex, "main"], cwd=paper_dir)
    run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=paper_dir)
    run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=paper_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", default=None, help="Path to paper/iclr")
    parser.add_argument("--limit", type=int, default=MAIN_PAGE_LIMIT, help="Main-text page limit")
    parser.add_argument(
        "--desktop-copy",
        default=str(DEFAULT_DESKTOP_PDF),
        help="Optional path for the versioned final PDF copy.",
    )
    parser.add_argument("--clean", action="store_true", help="Clean LaTeX intermediates after a successful build")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paper_dir = Path(args.paper_dir).resolve() if args.paper_dir else repo_root / "paper" / "iclr"
    main_tex = paper_dir / "main.tex"
    if not main_tex.exists():
        raise FileNotFoundError(main_tex)

    old_cwd = Path.cwd()
    try:
        os.chdir(paper_dir)
        try:
            build_with_latexmk(paper_dir)
        except FileNotFoundError:
            build_with_pdflatex(paper_dir)
    finally:
        os.chdir(old_cwd)

    aux_path = paper_dir / "main.aux"
    pdf_path = paper_dir / "main.pdf"
    if not pdf_path.exists():
        raise RuntimeError(f"Build finished without expected PDF: {pdf_path}")

    main_pages = parse_label_page(aux_path, "maintextend")
    total_pages = total_pdf_pages(pdf_path)
    print(f"Main-text page label before references: {main_pages}")
    if total_pages is not None:
        print(f"Total PDF pages: {total_pages}")
    print(f"PDF: {pdf_path}")

    if main_pages > args.limit:
        raise RuntimeError(
            f"Main text is {main_pages} pages, exceeding the ICLR initial-submission limit of {args.limit}."
        )

    if args.desktop_copy:
        desktop_copy = Path(args.desktop_copy).expanduser()
        if not desktop_copy.is_absolute():
            desktop_copy = repo_root / desktop_copy
        desktop_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, desktop_copy)
        print(f"Desktop PDF: {desktop_copy}")

    if args.clean:
        latexmk = shutil.which("latexmk")
        if latexmk:
            run([latexmk, "-c", "main.tex"], cwd=paper_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
