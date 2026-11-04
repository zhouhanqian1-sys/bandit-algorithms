"""Compile and validate the plain LaTeX Gradescope report without rerunning experiments.

Run from any directory with Python, pypdf, and pdflatex installed. The editable
source is report/BIOSTAT825_HW1_Report.tex; all charts are original saved PDFs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from pypdf import PdfReader

PROJECT = Path(__file__).resolve().parents[1]
ARCHIVE = PROJECT / "artifacts/full-20261101-1923"
SOURCE = PROJECT / "report/BIOSTAT825_HW1_Report.tex"
COMPARISONS = {
    "parts1-6": {
        "lin_experiment": "tab:part1", "epsilon_experiment": "tab:epsilon",
        "part2_experiment": "tab:part2", "part3_experiment": "tab:part3",
        "part4_linear_experiment": "tab:part4", "logistic_experiment": "tab:part4",
        "quadratic_experiment": "tab:part4", "quadratic_neural_results": "tab:part5",
        "shuttle_tuning_results": "tab:shuttle-tuning", "shuttle_results": "tab:shuttle-final",
    },
    "7a": {"movie_rewards": "tab:movies", "movie_rewards_b": "tab:movies"},
    "7b": {"medical_rewards_a": "tab:medical", "medical_rewards_b": "tab:medical"},
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output, engine):
    executable = shutil.which(engine)
    if executable is None:
        raise RuntimeError(f"{engine} is required to compile the LaTeX report.")
    source_text = SOURCE.read_text()
    figure_paths = [(SOURCE.parent / relative).resolve() for relative in
                    re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", source_text)]
    assert len(figure_paths) == len(set(figure_paths)) == 16
    assert all(path.is_file() for path in figure_paths)
    assert all(path.is_relative_to(ARCHIVE) for path in figure_paths)
    assert len(re.findall(r"\\section\*\{Part (?:[1-6]|7[AB])\.", source_text)) == 8
    previous_validation = json.loads((ARCHIVE / "validation-report.json").read_text())
    assert previous_validation["all_phases_validated"] and previous_validation["passed"]

    temp = PROJECT / "tmp/pdfs/latex"
    temp.mkdir(parents=True, exist_ok=True)
    command = [executable, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
               f"-output-directory={temp}", SOURCE.name]
    for iteration in range(3):
        result = subprocess.run(command, cwd=SOURCE.parent, capture_output=True, text=True)
        (temp / f"pass-{iteration + 1}.txt").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"LaTeX failed; inspect {temp / f'pass-{iteration + 1}.txt'}")
    log = (temp / f"{SOURCE.stem}.log").read_text(errors="replace")
    critical = re.findall(r"[^\n]*(?:Overfull \\[hv]box|Float too large|undefined references|"
                          r"Reference .+ undefined|Token not allowed in a PDF string)[^\n]*", log)
    if critical:
        raise RuntimeError("Resolve LaTeX layout/reference warnings:\n" + "\n".join(critical))

    aux = (temp / f"{SOURCE.stem}.aux").read_text()
    labels = {name: {"number": number, "page": int(page)} for name, number, page in
              re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{(\d+)\}", aux)}
    reader = PdfReader(temp / f"{SOURCE.stem}.pdf")
    texts = [page.extract_text() for page in reader.pages]
    full_text = "\n".join(texts)
    assert not reader.is_encrypted
    assert not any(character in full_text for character in ("\x00", "\ufffd", "\u25a0", "\u25a1"))
    assert not re.search(r"TODO|TBD|pending execution|[\u4e00-\u9fff]", full_text)
    figure_labels = {value["number"]: (name, value["page"]) for name, value in labels.items()
                     if name.startswith("fig:")}
    assert set(figure_labels) == {str(i) for i in range(1, 17)}
    checked_pairs = 0
    for phase, comparisons in COMPARISONS.items():
        summary = json.loads((ARCHIVE / phase / "summary.json").read_text())
        for comparison, label in comparisons.items():
            page = labels[label]["page"]
            text = re.sub(r"\s+", " ", texts[page - 1])
            for method, values in summary[comparison].items():
                mean, std = f"{values['final_mean']:.2f}", f"{values['final_sample_std']:.2f}"
                pattern = rf"(?<![\d.]){re.escape(mean)}\s*±\s*{re.escape(std)}(?![\d.])"
                assert re.search(pattern, text), (comparison, method, page, mean, std)
                assert values["T"] == 1000
                checked_pairs += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(temp / f"{SOURCE.stem}.pdf", output)
    page_starts = []
    for part in ("1", "2", "3", "4", "5", "6", "7A", "7B"):
        for index, text in enumerate(texts, 1):
            if re.search(rf"Part {part}\.\s", text):
                page_starts.append({"part": part, "page": index})
                break
    assert len(page_starts) == 8
    manifest = {
        "report": str(output.relative_to(PROJECT)), "report_sha256": digest(output),
        "format": "LaTeX article", "pages": len(reader.pages),
        "source_tex": str(SOURCE.relative_to(PROJECT)), "source_tex_sha256": digest(SOURCE),
        "source_notebook": "Experiments.ipynb", "source_notebook_sha256": digest(PROJECT / "Experiments.ipynb"),
        "source_execution_commit": "13f4e56dea345669e56e07da1dc06c2d2ea3f5ac",
        "assignment_sections": [item["part"] for item in page_starts], "page_starts": page_starts,
        "original_T": 1000, "original_N": {"1-2": 50, "3-4": 5, "5-6": 10, "7A-7B": 3},
        "vector_figures": [{"number": i, "page": figure_labels[str(i)][1],
                            "label": figure_labels[str(i)][0],
                            "source": str(path.relative_to(PROJECT)), "sha256": digest(path)}
                           for i, path in enumerate(figure_paths, 1)],
    }
    validation = {
        "passed": True, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_sha256": digest(output), "pages": len(reader.pages), "assignment_sections": 8,
        "vector_figures_verified_against_originals": 16,
        "algorithm_results_mean_and_sample_sd_pairs_verified": checked_pairs,
        "all_T_equal_1000": True, "latex_compilation_passes": 3,
        "overfull_boxes": 0, "oversize_floats": 0, "undefined_references": 0,
        "no_missing_or_null_glyphs": True, "no_placeholders_or_non_English_text": True,
        "visual_review": {"state": "pending rendered-page review"},
    }
    for name, value in (("report_manifest.json", manifest), ("report_validation.json", validation)):
        (output.parent / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"output": str(output), "pages": len(reader.pages),
                      "result_pairs_verified": checked_pairs, "page_starts": page_starts}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "output/pdf/BIOSTAT825_HW1_Report.pdf")
    parser.add_argument("--engine", default="pdflatex")
    args = parser.parse_args()
    build(args.output.resolve(), args.engine)


if __name__ == "__main__":
    main()
