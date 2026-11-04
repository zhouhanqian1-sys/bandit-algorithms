"""Build the English Gradescope report from the completed notebook and saved figures.

Authoring dependencies: reportlab and pypdf. No experiment is executed, and no
model or dataset is loaded. The existing chart PDFs are embedded as vector art.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape

from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate, Flowable, Frame, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)


PROJECT = Path(__file__).resolve().parents[1]
ARCHIVE = PROJECT / "artifacts/full-20261101-1923"
NAVY = colors.HexColor("#18334D")
TEAL = colors.HexColor("#137A83")
INK = colors.HexColor("#1F2933")
GRAY = colors.HexColor("#596775")
PALE = colors.HexColor("#F1F5F8")
LINE = colors.HexColor("#D6DFE6")
WIDTH = 516
CHART_WIDTH = 488
FONT_ROOT = Path("/System/Library/Fonts/Supplemental")
CELL_IDS = {
    "1": "7b3c9886-9d4d-4b30-8479-8adcaff5c36b", "2": "422e51d7",
    "3": "10f476dc", "4": "d3cd8521", "5": "55410b3e", "6": "04dcc399",
    "7A": "326edfde", "7B": "f94acffe",
}
FIGURES = [
    ("parts1-6", "cell_008_figure_001", "Part 1: linear contextual-bandit policies."),
    ("parts1-6", "cell_012_figure_002", "Part 2: epsilon-LinTS exploration-rule sweep."),
    ("parts1-6", "cell_012_figure_003", "Part 2: selected epsilon-LinTS and fresh baselines."),
    ("parts1-6", "cell_019_figure_004", "Part 3: linear-model LMCTS and comparison policies."),
    ("parts1-6", "cell_022_figure_005", "Part 4: shared arm geometry, linear reward function."),
    ("parts1-6", "cell_023_figure_006", "Part 4: shared arm geometry, logistic reward function."),
    ("parts1-6", "cell_024_figure_007", "Part 4: shared arm geometry, quadratic reward function."),
    ("parts1-6", "cell_034_figure_008", "Part 5: neural policies in the quadratic environment."),
    ("parts1-6", "cell_038_figure_009", "Part 6: four tested LMCTS configurations on shuttle."),
    ("parts1-6", "cell_038_figure_010", "Part 6: fresh comparison with the selected LMCTS setting."),
    ("7a", "cell_052_figure_001", "Part 7A: cumulative reward for Role A (space science)."),
    ("7a", "cell_053_figure_002", "Part 7A: mean movie-selection counts per run for Role A."),
    ("7a", "cell_056_figure_003", "Part 7A: cumulative reward for Role B (romantic musicals)."),
    ("7a", "cell_057_figure_004", "Part 7A: mean movie-selection counts per run for Role B."),
    ("7b", "cell_072_figure_001", "Part 7B: cumulative satisfaction reward for Profile A."),
    ("7b", "cell_075_figure_002", "Part 7B: cumulative satisfaction reward for Profile B."),
]


def ascii_punctuation(text):
    return text.translate(str.maketrans({
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u00a0": " ",
    }))


def inline(text):
    text = escape(ascii_punctuation(text).replace("$", ""))
    for tag in ("sub", "super"):
        text = text.replace(f"&lt;{tag}&gt;", f"<{tag}>").replace(f"&lt;/{tag}&gt;", f"</{tag}>")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r'<font face="Mono">\1</font>', text)
    return text


def register_fonts(font_root):
    mapping = {
        "Body": "Times New Roman.ttf", "BodyBold": "Times New Roman Bold.ttf",
        "BodyItalic": "Times New Roman Italic.ttf", "BodyBI": "Times New Roman Bold Italic.ttf",
        "Sans": "Arial.ttf", "SansBold": "Arial Bold.ttf", "SansItalic": "Arial Italic.ttf",
        "Mono": "Courier New.ttf",
    }
    for family, filename in mapping.items():
        path = font_root / filename
        if not path.exists():
            raise FileNotFoundError(f"Supply --font-dir containing {filename}: {font_root}")
        pdfmetrics.registerFont(TTFont(family, str(path)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="BodyItalic", boldItalic="BodyBI")
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="SansBold", italic="SansItalic", boldItalic="SansBold")


def make_styles():
    return {
        "body": ParagraphStyle("Body", fontName="Body", fontSize=11.1, leading=14.0,
                               textColor=INK, spaceAfter=8),
        "small": ParagraphStyle("Small", fontName="Sans", fontSize=9.1, leading=12,
                                textColor=GRAY, spaceAfter=7),
        "caption": ParagraphStyle("Caption", fontName="Sans", fontSize=8.5, leading=10.8,
                                  textColor=GRAY, spaceBefore=3, spaceAfter=8),
        "h1": ParagraphStyle("H1", fontName="SansBold", fontSize=18.5, leading=23,
                             textColor=NAVY, spaceAfter=12),
        "h2": ParagraphStyle("H2", fontName="SansBold", fontSize=11.4, leading=15,
                             textColor=TEAL, spaceBefore=5, spaceAfter=6),
        "title": ParagraphStyle("Title", fontName="SansBold", fontSize=28, leading=32,
                                textColor=NAVY, spaceAfter=10),
        "kicker": ParagraphStyle("Kicker", fontName="SansBold", fontSize=10, leading=14,
                                 textColor=TEAL, spaceAfter=12),
        "table": ParagraphStyle("Table", fontName="Sans", fontSize=9.3, leading=12,
                                textColor=INK),
        "table_head": ParagraphStyle("TableHead", fontName="SansBold", fontSize=9.1,
                                     leading=11.5, textColor=colors.white),
        "formula": ParagraphStyle("Formula", fontName="Body", fontSize=11.2, leading=16,
                                  alignment=TA_CENTER, textColor=INK, spaceAfter=9, spaceBefore=4),
    }


def analysis_blocks(notebook):
    lookup = {c["id"]: c for c in notebook["cells"]}
    result = {}
    for part, cell_id in CELL_IDS.items():
        markdown = "".join(lookup[cell_id]["source"])
        paragraphs, tables = [], []
        for block in re.split(r"\n\s*\n", markdown.strip()):
            if block.startswith("Briefly summarize"):
                continue
            if block.lstrip().startswith("|"):
                rows = [[item.strip() for item in line.strip().strip("|").split("|")]
                        for line in block.splitlines() if line.strip()]
                tables.append([row for row in rows if not all(re.fullmatch(r"[: -]+", col) for col in row)])
            else:
                block = re.sub(r"^\*\*Results: Part [^*]+\*\*\s*", "", block)
                paragraphs.append(" ".join(block.splitlines()))
        result[part] = {"paragraphs": paragraphs, "tables": tables}
    return result


@dataclass
class ChartPosition:
    page: int
    figure: int
    path: Path
    x: float
    y: float
    width: float
    height: float


class VectorChart(Flowable):
    def __init__(self, path, figure_number, positions, width=CHART_WIDTH):
        super().__init__()
        self.path = path
        self.figure_number = figure_number
        self.positions = positions
        box = PdfReader(path).pages[0].mediabox
        self.width = width
        self.height = width * float(box.height) / float(box.width)
        self.hAlign = "CENTER"

    def draw(self):
        x, y = self.canv.absolutePosition(0, 0)
        self.positions.append(ChartPosition(self.canv.getPageNumber(), self.figure_number,
                                            self.path, x, y, self.width, self.height))


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states = []

    def showPage(self):
        self.states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self.states)
        for state in self.states:
            self.__dict__.update(state)
            self.setFont("Sans", 8)
            self.setFillColor(GRAY)
            self.drawString(48, 23, "BIOSTAT825 Homework 1 | zq63")
            self.drawRightString(564, 23, f"{self._pageNumber} / {total}")
            super().showPage()
        super().save()


class ReportDocument(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(str(filename), pagesize=letter, leftMargin=48, rightMargin=48,
                         topMargin=60, bottomMargin=44, title="BIOSTAT825 Homework 1 - Experimental Report",
                         author="zq63", subject="Eight completed contextual and LLM bandit experiments")
        self.page_titles = []
        frame = Frame(48, 44, WIDTH, 688, leftPadding=0, rightPadding=0, topPadding=0,
                      bottomPadding=0, id="main")
        self.addPageTemplates(PageTemplate(id="report", frames=[frame], onPage=self.header))

    def header(self, canv, doc):
        canv.saveState()
        canv.setStrokeColor(LINE)
        canv.setLineWidth(0.6)
        canv.line(48, 751, 564, 751)
        canv.setFillColor(GRAY)
        canv.setFont("Sans", 8.1)
        canv.drawString(48, 763, "EXPERIMENTAL REPORT")
        canv.drawRightString(564, 763, "BIOSTAT825 | HOMEWORK 1")
        canv.restoreState()

    def afterFlowable(self, flowable):
        if hasattr(flowable, "report_title"):
            self.page_titles.append({"title": flowable.report_title, "page": self.page})


def build_report(output, font_dir):
    register_fonts(font_dir)
    styles = make_styles()
    notebook_path = PROJECT / "Experiments.ipynb"
    notebook = json.loads(notebook_path.read_text())
    results = analysis_blocks(notebook)
    validation = json.loads((ARCHIVE / "validation-report.json").read_text())
    assert validation["all_phases_validated"] and validation["passed"]
    assert sum(c["cell_type"] == "code" and c.get("execution_count") is not None
               for c in notebook["cells"]) == 43
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = PROJECT / "tmp/pdfs"
    temp.mkdir(parents=True, exist_ok=True)
    base_path = temp / "report-layout.pdf"
    story, positions, used_figure_sources = [], [], []

    def p(text, style="body"):
        return Paragraph(inline(text), styles[style])

    def add(text, style="body"):
        story.append(p(text, style))

    def title(text, new_page=True):
        if new_page:
            story.append(PageBreak())
        paragraph = p(text, "h1")
        paragraph.report_title = ascii_punctuation(text)
        story.append(paragraph)

    def table(rows, widths):
        cells = [[p(value, "table_head" if i == 0 else "table") for value in row]
                 for i, row in enumerate(rows)]
        t = Table(cells, colWidths=widths, hAlign="LEFT", repeatRows=1)
        commands = [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, -1), (-1, -1), 0.5, LINE),
        ]
        for row in range(1, len(rows)):
            if row % 2:
                commands.append(("BACKGROUND", (0, row), (-1, row), PALE))
        t.setStyle(TableStyle(commands))
        story.append(t)
        story.append(Spacer(1, 9))

    def result_table(part, index=0, widths=None):
        rows = results[part]["tables"][index]
        if widths is None:
            widths = [WIDTH * 0.60, WIDTH * 0.40]
        table(rows, widths)

    def chart(number, width=CHART_WIDTH, extra=""):
        phase, stem, caption = FIGURES[number - 1]
        used_figure_sources.append(number)
        display_number = len(used_figure_sources)
        figure = VectorChart(ARCHIVE / phase / "figures" / f"{stem}.pdf", display_number, positions, width)
        caption = f"**Figure {display_number}.** {caption}" + (" " + extra if extra else "")
        story.append(KeepTogether([figure, p(caption, "caption")]))

    def discussion(part, paragraph_indices):
        add("Results and discussion", "h2")
        for index in paragraph_indices:
            add(results[part]["paragraphs"][index])

    # Page 1: clear identification, study design, and interpretation conventions.
    add("HOMEWORK 1 / COMPLETED EXPERIMENTS", "kicker")
    add("Contextual and LLM Bandits", "title")
    add("BIOSTAT825 - Experimental Results and Discussion", "h2")
    add("Student NetID: **zq63** | Submission report: **November 3, 2026**", "small")
    add("This report presents the eight completed assignment sections in the order Parts 1-6, "
        "7A, and 7B. All results come from the executed notebook and saved full-size arrays. "
        "The original horizon is **T = 1000** throughout. No experiment was rerun or reduced "
        "to prepare this report.")
    table([
        ["Assignment section", "Monte Carlo runs N", "Report pages"],
        ["Part 1 - Linear contextual policies", "50", "2"],
        ["Part 2 - Epsilon-LinTS selection", "50 per comparison", "3-4"],
        ["Part 3 - Linear-model LMCTS", "5", "5"],
        ["Part 4 - Three reward functions", "5 per environment", "6-7"],
        ["Part 5 - Neural quadratic bandits", "10", "8"],
        ["Part 6 - Shuttle and LMCTS tuning", "10 per comparison", "9-10"],
        ["Part 7A - LLM movie recommendations", "3 per role", "11-13"],
        ["Part 7B - LLM patient preferences", "3 per profile", "14-15"],
    ], [284, 128, 104])
    add("Evaluation conventions", "h2")
    add("Parts 1-6 measure cumulative **pseudo-regret** from expected rewards; lower is better. "
        "Parts 7A-7B measure cumulative sampled reward; higher is better.")
    story.append(Paragraph("R(T) = Σ<sub>t=1</sub><super>T</super> [max<sub>a</sub> μ<sub>t</sub>(a) "
                           "- μ<sub>t</sub>(a<sub>t</sub>)] &nbsp;&nbsp;&nbsp; "
                           "C(T) = Σ<sub>t=1</sub><super>T</super> r<sub>t</sub>", styles["formula"]))
    add("Tables report final totals as **mean ± sample standard deviation**. Shaded curve bands "
        "are pointwise empirical 10th-90th percentiles across runs, not confidence intervals. "
        "Policies start each run with fresh state and use matched environment randomness within "
        "each comparison. Rankings describe the tested instances and seeds.")
    add("Execution: Duke DCC, one H200, existing RL environment; job 60876090 completed in 2:01:18. "
        "All 43 code cells and 16 saved figure pairs passed independent result validation. "
        "Reproducibility details and model revisions are on page 16.", "small")

    # Page 2.
    title("Part 1. Linear contextual bandits")
    add("The synthetic linear environment has d = 7, three available arms per round, and two "
        "fixed context sets. Arm vectors and the unknown coefficient have unit norm. "
        "Observed rewards are x<super>T</super>θ plus Gaussian noise with standard deviation 1. "
        "Greedy, epsilon-greedy (0.1), LinUCB, and LinTS use ridge/prior precision 1.")
    chart(1)
    result_table("1")
    discussion("1", [1])

    # Pages 3-4.
    title("Part 2. Choosing epsilon for LinTS")
    add("Epsilon-LinTS uses a Thompson draw with probability epsilon and the greedy ridge "
        "prediction otherwise. The four tested rules are 0.05, 1/d = 1/7, 1/t, and an "
        "additional constant 0.2. Time is one-based, so 1/t starts at 1. The sweep reuses "
        "the Part 1 environment with N = 50, T = 1000, and seed 62.")
    chart(2)
    result_table("2", 0)
    add("Selection and stability", "h2")
    add(results["2"]["paragraphs"][1].rstrip(":") + ".")
    add("The selected 1/t rule has the lowest observed sweep mean but a standard deviation "
        "of 58.04. By comparison, 1/d has a slightly higher mean (18.56) with a much smaller "
        "standard deviation (12.84). Mean-based selection therefore trades average performance "
        "against sensitivity to unfavorable trajectories.")

    title("Part 2. Fresh baseline comparison")
    add("The selected 1/t rule is compared with fresh baseline agents and trajectory seed 63, "
        "again using N = 50 and T = 1000. No sweep trajectory is reused as a fresh-comparison result.")
    chart(3)
    result_table("2", 1)
    discussion("2", [2])

    # Page 5.
    title("Part 3. Linear-model LMCTS")
    add("A new linear environment uses N = 5 and T = 1000. LMCTS selects the largest current "
        "prediction from a bias-free linear model and makes one warm-start Langevin update "
        "per observation using all available history. Settings are learning rate 0.1, inverse "
        "temperature 0.01, noise multiplier 1, and weight decay 1. The loss is mean MSE.")
    chart(4)
    result_table("3")
    discussion("3", [1])
    add("These finite-step, regularized-loss Langevin updates are the notebook's explicit "
        "implementation assumption; an exact posterior or convergence guarantee is not claimed.", "small")

    # Pages 6-7: keep the first two charts large enough for readable legends.
    title("Part 4. Linear and logistic rewards")
    add("All three environments share the same six arm-feature vectors and coefficient, with "
        "N = 5 and T = 1000. The mean is x<super>T</super>θ for linear rewards, "
        "sigmoid(x<super>T</super>θ) for Bernoulli logistic rewards, and "
        "10(x<super>T</super>θ)<super>2</super> for Gaussian quadratic rewards. Policies and "
        "LMCTS settings are the same as in Part 3.")
    chart(5, width=476)
    chart(6, width=476)

    title("Part 4. Quadratic rewards and comparison")
    chart(7)
    result_table("4", 0, [156, 113, 113, 134])
    discussion("4", [1])

    # Page 8.
    title("Part 5. Neural policies for quadratic rewards")
    add("Each policy uses an independent scalar-output MLP with one 50-unit LeakyReLU hidden "
        "layer. Epsilon-greedy (0.1), NeuralTS, and NeuralUCB use SGD learning rate 0.01; "
        "updates use up to 64 historical observations and five steps. NeuralTS/UCB use a "
        "diagonal gradient-precision approximation with nu = lambda = 1. NeuralLMCTS uses "
        "learning rate 0.01, inverse temperature 0.01, noise multiplier 1, and decay 0.01.")
    chart(8)
    result_table("5")
    discussion("5", [1])

    # Pages 9-10.
    title("Part 6. LMCTS tuning on shuttle")
    add("The shuttle classification problem becomes a seven-arm contextual bandit: each "
        "nine-dimensional, row-normalized context occupies one of seven disjoint feature blocks "
        "(63 inputs). Correct choices earn 1 and other choices earn 0. The four-setting sweep "
        "uses seed 606, N = 10, and T = 1000, holding width 50, batch size 64, noise multiplier "
        "1, and weight decay 0.01 fixed.")
    chart(9)
    result_table("6", 0, [126, 79, 108, 64, 139])
    discussion("6", [1])

    title("Part 6. Fresh neural-policy comparison")
    add("The selected configuration (learning rate 0.01, inverse temperature 0.001, five updates) "
        "is evaluated against the other neural policies using fresh trajectory seed 607, "
        "N = 10, and T = 1000. This uses new online trajectories on the same dataset.")
    chart(10)
    result_table("6", 1, [216, 152, 148])
    discussion("6", [2])

    # Pages 11-13.
    title("Part 7A. LLM movie recommendations")
    add("Qwen3.5-0.8B scores one of eight movies without conversation history. Role A favors "
        "space science and engineering; Role B favors romantic musicals. An integer score/10 "
        "receives Gaussian noise with standard deviation 0.1 and is clipped to [0, 1]. "
        "Sampling temperature is 0.5. Random, Greedy, epsilon-greedy (0.1), and UCB use "
        "N = 3 and T = 1000 per role, with seed 7001. Bands are descriptive percentiles.")
    chart(11)
    chart(13)

    title("Part 7A. Rewards and Role A selections")
    add("Final cumulative rewards are mean ± sample standard deviation; higher is better.", "small")
    result_table("7A", 0, [228, 144, 144])
    chart(12)
    add(results["7A"]["paragraphs"][1])

    title("Part 7A. Role B selections and variability")
    chart(14)
    result_table("7A", 1, [106, 205, 205])
    add(results["7A"]["paragraphs"][2])
    add("UCB uses the bonus sqrt(2 log(t + 1) / n) and tries every movie once. Exact ties "
        "are broken randomly. Identical choices share a fresh LLM score within that round; "
        "later rounds always obtain new scores.", "small")

    # Pages 14-15.
    title("Part 7B. Contextual LLM patient preferences")
    add("Five fixed ChatDoctor questions provide 17 distinct candidate responses from "
        "UltraMedical-Preference. Qwen2.5-0.5B last-token embeddings undergo a seed-42 random "
        "projection to 32 dimensions, normalization, centering/PCA to 16 components, common-radius "
        "scaling, and intercept addition, yielding 17 features. Qwen3.5-0.8B supplies score/10 "
        "rewards without added Gaussian noise (temperature 0.3). Profile A favors plain main "
        "advice and a clear next step; Profile B favors relevant detailed reasoning.")
    chart(15)
    add("Final results: N = 3 and T = 1000 per profile", "h2")
    result_table("7B", 0, [116, 132, 68, 132, 68])
    add(results["7B"]["paragraphs"][1])

    title("Part 7B. Profile B and interpretation")
    chart(16)
    add("Model and comparison design", "h2")
    add("LinUCB uses alpha = lambda = 1. LinTS samples a shared linear coefficient with "
        "nu = lambda = 1. Each learning algorithm maintains its own ridge estimate across "
        "questions. All policies receive the same question in a given round; identical response "
        "choices share a freshly sampled score. Seed 7002 gives matching A/B question schedules. "
        "The Gaussian linear model approximates the bounded satisfaction reward.")
    add("Results and discussion", "h2")
    add(results["7B"]["paragraphs"][2])
    add(results["7B"]["paragraphs"][3])

    # Page 16: interpretation assumptions and exact source provenance.
    title("Reproducibility and modeling assumptions")
    add("Algorithm conventions", "h2")
    add("Linear TS uses a zero-mean Gaussian prior with precision 1 and unit observation-noise "
        "variance, drawing N(theta-hat, V-inverse). The supplied LinUCB radius is retained as a "
        "heuristic. LMCTS uses the noisy regularized-loss update below, with z standard normal.")
    story.append(Paragraph("θ ← θ - η(grad MSE(θ) + λθ) + σ sqrt(2ηβ<super>-1</super>)z", styles["formula"]))
    add("For NeuralTS/UCB, normalized gradient features are phi = gradient(f)/sqrt(50), and "
        "the diagonal precision accumulates squared features. Training uses mean MSE plus "
        "lambda times 50 times the squared distance from initial weights divided by the "
        "observation count. These neural uncertainty and Langevin conventions are explicit "
        "implementation assumptions, not exact posterior guarantees.")
    add("Seeds and saved execution", "h2")
    table([
        ["Part", "Environment / trajectory seeds"],
        ["1-2", "42 / 52; epsilon sweep 62; fresh comparison 63"],
        ["3-4", "72 / 73; Part 4 coefficient 82, geometry 83, trajectories 84"],
        ["5-6", "Part 5: 505; shuttle tuning 606; fresh comparison 607"],
        ["7A-7B", "Movie roles: 7001; patient profiles: 7002"],
    ], [90, 426])
    add("Experiment source: **13f4e56dea345669e56e07da1dc06c2d2ea3f5ac**. "
        "Run archive: artifacts/full-20261101-1923. All 43 executed cells, original N/T, "
        "finite arrays, saved statistics, and 16 PNG/PDF pairs passed independent validation. "
        "Actual reward generation completed 18,479 movie and 12,757 patient calls with zero failures.", "small")
    add("Pinned sources and software", "h2")
    table([
        ["Source", "Pinned revision"],
        ["Qwen/Qwen3.5-0.8B (both scorers)", "2fc06364715b967f1860aea9cf38778875588b17"],
        ["Qwen/Qwen2.5-0.5B (encoder)", "060db6499f32faf8b98477b0a26969ef7d8b9987"],
        ["TsinghuaC3I/UltraMedical-Preference", "761eb7935310ba662a96d93c5af342e5269d5759"],
    ], [223, 293])
    add("RL environment: Python 3.11.18; PyTorch 2.10.0+cu128; NumPy 2.4.6; pandas 3.0.6; "
        "scikit-learn 1.9.1; transformers 5.3.0; datasets 4.8.5. Shuttle was loaded through "
        "OpenML's active dataset version. Both LLM scorers used top-p 0.9 and top-k 20.", "small")
    add("The companion code submission is Experiments.ipynb. Source and archived results: "
        "https://github.com/zhouhanqian1-sys/bandit-algorithms. The report uses existing "
        "notebook analyses and original vector figures; no raw medical response texts or model "
        "cache files are included.", "small")

    doc = ReportDocument(base_path)
    doc.build(story, canvasmaker=NumberedCanvas)
    reader = PdfReader(base_path)
    writer = PdfWriter()
    writer.append_pages_from_reader(reader)
    for chart_position in positions:
        original = PdfReader(chart_position.path).pages[0]
        box = original.mediabox
        scale = chart_position.width / float(box.width)
        transform = Transformation().scale(scale).translate(
            chart_position.x - float(box.left) * scale,
            chart_position.y - float(box.bottom) * scale,
        )
        writer.pages[chart_position.page - 1].merge_transformed_page(original, transform)
    writer.add_metadata({"/Title": "BIOSTAT825 Homework 1 - Experimental Report",
                         "/Author": "zq63", "/Subject": "Eight full-size bandit experiments with results and discussion"})
    for item in doc.page_titles:
        writer.add_outline_item(item["title"], item["page"] - 1)
    writer.write(output)
    manifest = {
        "report": str(output.relative_to(PROJECT)), "pages": len(reader.pages),
        "report_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "assignment_sections": list(CELL_IDS), "source_notebook": "Experiments.ipynb",
        "source_notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        "source_execution_commit": "13f4e56dea345669e56e07da1dc06c2d2ea3f5ac",
        "original_T": 1000, "page_starts": doc.page_titles,
        "vector_figures": [{"number": pos.figure, "page": pos.page,
                            "source": str(pos.path.relative_to(PROJECT)),
                            "sha256": hashlib.sha256(pos.path.read_bytes()).hexdigest()}
                           for pos in positions],
    }
    (output.parent / "report_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "pages": len(reader.pages),
                      "figures": len(positions), "page_starts": doc.page_titles}, indent=2))
    assert len(reader.pages) == 16, "Unexpected automatic page break; inspect and adjust the layout."
    assert sorted(pos.figure for pos in positions) == list(range(1, 17))
    assert [item["page"] for item in doc.page_titles] == list(range(2, 17))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "output/pdf/BIOSTAT825_HW1_Report.pdf")
    parser.add_argument("--font-dir", type=Path, default=FONT_ROOT)
    args = parser.parse_args()
    build_report(args.output.resolve(), args.font_dir)


if __name__ == "__main__":
    main()
