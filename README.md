# BIOSTAT825 bandit experiments

The completed [Experiments.ipynb](Experiments.ipynb) includes all eight assignment sections, recorded outputs, and English result analyses. The original horizons and replication counts were used on one Duke DCC H200 in the existing `RL` environment, in the order Parts 1–6, 7A, and 7B. Slurm job `60876090` completed successfully in 2:01:18 using account and partition `scavenger-h200`.

For submission, upload the [English experimental report (PDF)](output/pdf/BIOSTAT825_HW1_Report.pdf) to Gradescope and [Experiments.ipynb](Experiments.ipynb) to the Canvas Assignment section. The report uses a plain LaTeX article layout and covers Parts 1–6, 7A, and 7B, with all 16 original vector figures, numerical tables, methods, and English discussion. The editable [LaTeX source](report/BIOSTAT825_HW1_Report.tex) is included for local revision; the instructor does not require it for submission.

Run `python scripts/build_submission_pdf.py` to compile the report from the project checkout, using pdflatex and pypdf without rerunning experiments. The source includes the original saved chart PDFs. See [report_manifest.json](output/pdf/report_manifest.json) for source hashes and page locations, [report_validation.json](output/pdf/report_validation.json) for content and layout checks, and [report_revision.json](output/pdf/report_revision.json) for the concise writing revision record based on Nature Skills guidance.

The [full result archive](artifacts/full-20261101-1923/) contains 16 PNG/PDF figure pairs, numerical NPZ arrays, JSON summaries, phase logs, raw execution snapshots, and the combined analyzed notebook. See the [validation report](artifacts/full-20261101-1923/validation-report.json) and [run metadata](artifacts/full-20261101-1923/run-metadata.json) for exact parameters, pinned revisions, environment versions, and provenance. Experiment code was executed from commit `13f4e56dea345669e56e07da1dc06c2d2ea3f5ac`; subsequent changes add validation, recorded outputs, and interpretation.

To check saved results without rerunning the experiments:

```bash
python scripts/validate_full_results.py --output-dir artifacts/full-20261101-1923
python tests/validate_notebook.py --allow-executed
```

The archive preserves the original N/T: T=1000 throughout; N=50 for Parts 1–2, N=5 for Parts 3–4, N=10 for Parts 5–6, and N=3 per role/profile for 7A–7B. Mean ± sample standard deviation and 10th–90th percentile bands summarize the observed trajectories; the bands are not confidence intervals. Model caches, full raw datasets, private SSH files, and recovery checkpoints are excluded from version control.

# On local machine
Please make sure you have python environment on your local machine.

## Install the required packages

```
pip install -r requirements.txt
```

# On Google Colab

## Upload the files in google drive

1. Please click https://workspace.google.com/products/drive/ and sign in with your google account.
2. Upload the files in google drive.
3. Open the Experiments.ipynb file in google colab.
4. Mount the google drive in google colab by running the following code:

```
from google.colab import drive
drive.mount('/content/drive')
```

5. Change the working directory to the folder where you uploaded the files by running the following code:

```
PROJECT_ROOT = "/content/drive/MyDrive/your_folder_name"  # change to your folder name
import os, sys
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
```

6. Install the required packages

```
!pip install -r requirements.txt
```
