# GROUP25_CENG313 — Multilingual Chatbot Arena: Human Preference Prediction

CENG313 term project. We take part (late submission) in the Kaggle competition
**[WSDM Cup – Multilingual Chatbot Arena](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena)**.

## Team

| Name | Student ID | GitHub |
|------|-----------|--------|
| Cemil Emre Bıyık | 24118080048 | [@shroudy-bit32](https://github.com/shroudy-bit32) |
| Ömer Berkay Pısdıl | 24118080090 | [@CiciMuz](https://github.com/CiciMuz) |

## Competition

Users of [Chatbot Arena](https://lmarena.ai) write a prompt, get answers from two anonymous LLMs
and vote for the better one. Given `prompt`, `response_a`, `response_b` we must predict the
`winner` (`model_a` or `model_b`). Conversations are in many languages.

* **Metric:** classification accuracy
* **Submission:** `submission.csv` with columns `id,winner`
* It is a *code competition*: the real test set is hidden and is only swapped in when the Kaggle notebook is re-run on submit.

## Dataset

| File | Description |
|------|-------------|
| `train.parquet` | 48,439 battles: `id, prompt, response_a, response_b, winner, model_a, model_b, language` (128 languages, 60 LLMs) |
| `test.parquet` | Tiny public sample of the hidden test set (same columns, no labels / model names) |
| `sample_submission.csv` | Submission format |

Data files are **not** committed (`train.parquet` is ~114 MB, above GitHub's limit). Download them with

```bash
kaggle competitions download -c wsdm-cup-multilingual-chatbot-arena
unzip wsdm-cup-multilingual-chatbot-arena.zip -d data/
```

## Repository layout

```
GROUP25_CENG313/            <- reusable python package used by the notebooks
    data.py                 loading (local / Colab / Kaggle), target encoding, submission writer
    features.py             language-agnostic handcrafted features (length, markdown, overlap, ...)
    models.py               baseline models with a common fit / predict_proba interface
    evaluation.py           stratified K-fold CV, result tables, per-group accuracy
    eda.py                  plotting helpers for the EDA
notebooks/
    Assignment1_EDA_Baseline.ipynb     Assignment 1 – EDA + baselines (Colab)
    kaggle_submission_baseline.ipynb   Kaggle inference notebook that produces submission.csv
requirements.txt
```

## Approach

### Assignment 1 — EDA and baselines

Key findings (details in the notebook):

* Balanced target (model_b wins 50.5%), 128 languages (English 52%, Russian 13%, Chinese 9%, Vietnamese 6%).
* **Verbosity bias:** the longer answer wins 58.4% of the battles.
* **Formatting bias:** more headers / bold / lists / code blocks → higher win rate. Refusals win only 26%.
* Model names (strong predictors) exist only in train, not in test.

Baselines (all evaluated with 5-fold stratified CV on the training set):

1. **Majority class** – always predict the most frequent label.
2. **Longer response wins** – rule-based, exploits verbosity bias.
3. **Handcrafted features + Logistic Regression** – ~90 features per pair (lengths, markdown structure,
   refusals, prompt overlap …) as A/B values, differences and ratios.
4. **Handcrafted features + LightGBM** – same features, gradient boosting.
5. **Char n-gram TF-IDF difference + Logistic Regression** – `tfidf(A) - tfidf(B)` with
   character n-grams (works for every script), trained with A/B swap augmentation.

### Assignment 2 — proposed solution

_To be added._

## Results

5-fold stratified cross-validation on the training set (accuracy):

| Model | CV accuracy |
|---|---|
| Majority class | 0.5054 |
| Longer response wins (rule) | 0.5837 |
| Handcrafted features + LogReg | 0.6039 ± 0.0025 |
| Char TF-IDF diff + LogReg | 0.6046 ± 0.0044 |
| Handcrafted features + LightGBM | 0.6128 ± 0.0028 |
| **Blend: LightGBM + TF-IDF** | **0.6271** |

## How to run

**Google Colab:** open a notebook from `notebooks/` in Colab (File → Open notebook → GitHub). The first cell
clones this repository, installs the requirements and downloads the data with the Kaggle API
(upload your `kaggle.json` when asked, or put the parquet files in `MyDrive/wsdm-cup-multilingual-chatbot-arena/`).

**Locally:**

```bash
pip install -r requirements.txt
# put train.parquet / test.parquet into ./data
jupyter notebook notebooks/Assignment1_EDA_Baseline.ipynb
```

**Kaggle submission:** create a Kaggle notebook from `notebooks/kaggle_submission_baseline.ipynb`, add the
competition data, upload the `GROUP25_CENG313/` folder as a Kaggle dataset and attach it, turn
internet **off**, then *Submit to competition*.

## Kaggle submission history

| Date | Notebook / model | Public LB | Private LB |
|------|------------------|-----------|------------|
| | | | |

## Progress log

| Date | Work done |
|------|-----------|
| 2026-11-04 | Repository created; data downloaded; package skeleton, EDA and baseline models (Assignment 1) |
