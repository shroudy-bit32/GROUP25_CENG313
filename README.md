# GROUP25_CENG313 — Multilingual Chatbot Arena: Human Preference Prediction

**Group ID:** 25 · CENG313 Introduction to Data Science

Kaggle competition: **[WSDM Cup – Multilingual Chatbot Arena](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena)**
(closed competition → evaluated with **late submissions**)

## Team

| Name | Student ID | GitHub |
|------|-----------|--------|
| Cemil Emre Bıyık | 24118080048 | [@shroudy-bit32](https://github.com/shroudy-bit32) |
| Ömer Berkay Pısdıl | 24118080090 | [@CiciMuz](https://github.com/CiciMuz) |

## Current Kaggle score

| Model | Validation acc. (20 % split) | 5-fold CV acc. | Kaggle public | Kaggle private | ≈ rank (private LB) |
|---|---|---|---|---|---|
| Blend LightGBM + char TF-IDF (Assignment 1 baseline) | 0.6216 | 0.6271 | – | **0.6224** | ~586 |

Screenshot: [`submissions/kaggle_late_submission.png`](submissions/kaggle_late_submission.png)

## Competition

Users of [Chatbot Arena](https://lmarena.ai) write a prompt, get answers from two anonymous LLMs and vote for the
better one. Given `prompt`, `response_a`, `response_b` we predict the `winner` (`model_a` or `model_b`).
Conversations are in 128 languages.

* **Target:** `winner` (binary) · **Metric:** classification accuracy · **Submission:** `submission.csv` with `id,winner`
* *Code competition*: the real test set is hidden and only swapped in when the Kaggle notebook is re-run on submit.

## Data

| File | Description |
|------|-------------|
| `train.parquet` | 48,439 battles: `id, prompt, response_a, response_b, winner, model_a, model_b, language` |
| `test.parquet` | public sample (3 rows) of the hidden test set: `id, prompt, response_a, response_b, scored` |
| `sample_submission.csv` | submission format |

Data files are **not committed** (Kaggle rules, `train.parquet` is ~114 MB). Download them into `data/`:

```bash
pip install kaggle            # needs ~/.kaggle/kaggle.json (Kaggle → Settings → API → Create New Token)
kaggle competitions download -c wsdm-cup-multilingual-chatbot-arena -p data
cd data && unzip wsdm-cup-multilingual-chatbot-arena.zip
```

## Repository structure

```
GROUP25_CENG313/
├── README.md
├── requirements.txt
├── .gitignore                         excludes data/
├── assignment1_eda_baseline.ipynb     Assignment 1: EDA & baseline (run with outputs)
├── assignment2_solution.ipynb         Assignment 2: improved solution (to be added)
├── src/
│   ├── __init__.py
│   ├── data.py                        loading, target encoding, train/valid split, handcrafted features, char n-grams
│   ├── models.py                      baseline models, hold-out evaluation, cross-validation
│   └── utils.py                       data-overview tables, EDA plots, saving submissions
├── kaggle/
│   └── kaggle_submission.ipynb        Kaggle notebook (internet off) that produces submission.csv
├── data/                              NOT committed
└── submissions/                       Kaggle score screenshots
```

## How to run

```bash
pip install -r requirements.txt
# download the data into ./data (see above)
jupyter notebook assignment1_eda_baseline.ipynb      # run from the repository root
```

The notebook imports the reusable code with `from src.data import ...`, `from src.models import ...`.
Running all cells takes ~10 minutes on a laptop CPU.

**Google Colab:** open the notebook from GitHub; the first cell clones the repository (private repo → use a GitHub token
in the URL) and installs the requirements. Put the data into `data/` (Kaggle API as above).

**Kaggle submission:** upload the `src/` folder as a Kaggle dataset, import `kaggle/kaggle_submission.ipynb` into a
Kaggle notebook, add the competition data and the `src` dataset as inputs, turn internet **off**, *Save Version*, then
*Submit to competition*.

## Approach

### Assignment 1 — EDA & baseline

EDA findings:
* balanced target (model_b wins 50.5 %), no missing values, practically no duplicates;
* **verbosity bias**: the longer answer wins 58.4 % of the battles;
* **formatting bias**: more headers / code blocks / bold text → higher win rate; refusals lose ~77 %;
* `model_a`, `model_b`, `language` are not available in the test set → not used as features.

Preprocessing: 95 language-agnostic handcrafted features (A, B, A−B, A/(A+B) of length & markdown statistics,
refusal flags, prompt overlap) and hashed character 2–4-gram TF-IDF (`tfidf(A) − tfidf(B)`, A/B swap augmentation);
`StandardScaler` for logistic regression; stratified 80/20 split (`random_state=42`).

| Model | Validation acc. | 5-fold CV acc. |
|---|---|---|
| Majority class | 0.5054 | 0.5054 |
| Longer response wins (rule) | 0.5873 | 0.5837 |
| **Handcrafted + Logistic Regression** (simple baseline) | 0.6114 | 0.6039 |
| Char TF-IDF diff + Logistic Regression | 0.5931 | 0.6046 |
| Handcrafted + LightGBM | 0.6182 | 0.6128 |
| **Blend LightGBM + TF-IDF** | **0.6216** | **0.6271** |

### Assignment 2 — proposed solution

_To be added_ (planned: fine-tuned multilingual transformer + stacking with the handcrafted features).

## Kaggle submission history

Late submissions (not ranked; "≈ rank" is where the score would be placed on the final private leaderboard).

| Date | Account | Notebook / model | CV acc. | Public | Private | ≈ Rank |
|------|---------|------------------|---------|--------|---------|--------|
| 2026-11-10 | Ömer Berkay Pısdıl | `kaggle/kaggle_submission.ipynb` – LightGBM + char TF-IDF blend | 0.6271 | – | 0.6224 | ~586 |

## Progress log

| Date | Work done |
|------|-----------|
| 2026-11-04 | Repository created, data downloaded, project setup and data loading utilities |
| 2026-11-05 | Team info added |
| 2026-11-06 | EDA helpers; handcrafted features and baseline models |
| 2026-11-08 | EDA notebook |
| 2026-11-09 | Cross-validation utilities; baseline experiments; Kaggle submission notebook |
| 2026-11-10 | Fixed Kaggle notebook paths; first Kaggle late submission (private 0.6224) |
| 2026-11-12 | Moved to the course organization; code reorganized into `src/`; notebook rewritten with the required sections |
| 2026-11-13 | README, Kaggle screenshot, Assignment 1 tag |
