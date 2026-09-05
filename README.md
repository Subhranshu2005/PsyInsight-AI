# 🧠 PsyInsight AI

> An Open-Source AI Platform for Psychological Research and Data Science

---

## 📖 Overview

PsyInsight AI is an intelligent research platform designed for psychology students and researchers. It combines:

- 📊 Data Science
- 📈 Statistics
- 🎲 Probability
- 🤖 Machine Learning
- 🧠 Deep Learning
- 🔤 Transformers (NLP)
- ⚙️ Automata Theory
- 💡 Explainable AI
- 📝 Automated Report Generation

into one unified research assistant, accessible either as a Python library or through a full Streamlit web application.

---

## 🚀 Quick Start

```bash
git clone https://github.com/Subhranshu2005/PsyInsight-AI.git
cd PsyInsight-AI

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -e ".[all]"              # everything, including torch/transformers/streamlit
# or a lighter install:
# pip install -e .                   # core only (numpy/pandas/scipy/sklearn)

streamlit run app/main.py            # launch the full research assistant UI
```

The app ships with a bundled sample dataset (`datasets/sample_psychology_data.csv`) so you can explore every tab immediately — click **"Load bundled sample dataset"** in the sidebar.

### Library usage

```python
from psyinsight.preprocessing import DataLoader, DataValidator, DataCleaner
from psyinsight.statistics.descriptive import DescriptiveStatistics
from psyinsight.statistics.inferential import InferentialStatistics
from psyinsight.probability.distributions import ProbabilityEngine
from psyinsight.ml import PsyClassifier, PsyRegressor, PsyClustering, PsyFeatureSelector, PsyMetrics
from psyinsight.xai import PsyExplainer
from psyinsight.automata import ResponsePatternAnalyzer
from psyinsight.transformers import TextAnalyzer
from psyinsight.reporting import ResearchReportGenerator

df = DataLoader.load("datasets/sample_psychology_data.csv")
print(DataValidator(df).validate())
print(DescriptiveStatistics(df).summary())
```

---

## 🧩 Module Guide

| Module | Path | What it does |
|---|---|---|
| **Preprocessing** | `psyinsight.preprocessing` | Loading (CSV/Excel/JSON), validation, cleaning |
| **Statistics** | `psyinsight.statistics` | Descriptive stats, t-tests, correlation, chi-square, ANOVA |
| **Probability** | `psyinsight.probability` | Normal/Binomial/Poisson distributions (SciPy + from-scratch), probability interpretation |
| **Machine Learning** | `psyinsight.ml` | `PsyClassifier`, `PsyRegressor`, `PsyClustering` (AutoML, algorithm registries), `PsyMetrics`, `PsyModelSelector`, `PsyFeatureSelector` |
| **Deep Learning** | `psyinsight.dl` | PyTorch MLP / LSTM / tiny-Transformer models + a generic `Trainer` (optional: requires `torch`) |
| **Transformers / NLP** | `psyinsight.transformers` | `TextAnalyzer`: sentiment, zero-shot theming, embeddings, theme discovery, keyword extraction (optional: requires `transformers`, falls back to a lexicon/TF-IDF mode otherwise) |
| **Automata Theory** | `psyinsight.automata` | DFA/NFA engines applied to survey-quality checks: Likert validity, straight-lining detection, participant-ID validation, pattern matching |
| **Explainable AI** | `psyinsight.xai` | `PsyExplainer`: global/permutation importance, partial dependence, LIME-style local explanations, optional SHAP |
| **Visualization** | `psyinsight.visualization` | Histograms, box plots, scatter/line/bar plots, correlation heatmaps |
| **Reporting** | `psyinsight.reporting` | `ResearchReportGenerator`: assembles every module's output into Markdown / HTML / JSON |
| **Utils** | `psyinsight.utils` | Logging, seeding, config, DataFrame helpers, custom exceptions |

Every module works standalone — the Streamlit app in `app/main.py` is simply a UI on top of the same library.

---

## 🖥️ The App

`streamlit run app/main.py` opens a tabbed workspace:

1. **Preprocessing** — upload/validate/clean your dataset
2. **Statistics** — descriptive stats, hypothesis tests, Cronbach's alpha
3. **Probability** — Normal/Binomial/Poisson calculators
4. **Machine Learning** — AutoML classification, regression, and clustering
5. **Feature Selection** — variance/correlation filters, importance ranking, PCA
6. **Explainable AI** — global & local explanations for your trained model
7. **Text / NLP** — sentiment, keyword extraction, theme discovery on open-ended responses
8. **Automata** — Likert-scale validity and careless-response (straight-lining) detection
9. **Visualization** — quick charts
10. **Report** — export everything you've flagged as a Markdown/HTML/JSON research report

---

## 🧪 Tests

```bash
pip install -e ".[dev]"
pytest
```

Deep-learning tests are automatically skipped if `torch` isn't installed; the NLP module always has a dependency-free fallback so its tests never require `transformers`.

---

## 💻 Tech Stack

- Python, NumPy, Pandas, SciPy, Scikit-learn
- PyTorch (deep learning)
- Hugging Face Transformers (NLP)
- Streamlit (app)
- Git & GitHub

---

## 📅 Status

**Version 0.2** — all planned modules implemented:

- [x] Data Preprocessing
- [x] Statistics Engine
- [x] Probability Engine
- [x] Machine Learning (classification, regression, clustering, feature selection, model selection, metrics)
- [x] Deep Learning
- [x] Transformers / NLP
- [x] Automata Theory
- [x] Visualization
- [x] Explainable AI
- [x] Report Generator
- [x] Full Streamlit application

---

⭐ This project is developed as an open-source portfolio and research project.
