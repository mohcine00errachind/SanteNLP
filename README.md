# SantéNLP

An NLP pipeline for English medical reports:

1. **Entity extraction**: finds SYMPTOM and TREATMENT mentions (spaCy NER, compared with a dictionary baseline).
2. **Specialty classification**: predicts the medical specialty of a report (fine-tuned Bio_ClinicalBERT, compared with a TF-IDF + logistic regression baseline).
3. **Streamlit app**: paste a report, see highlighted entities and the predicted specialty.

> Academic project. Uses only public data. **Not for clinical use.**

## Status

| Phase | State |
|---|---|
| 0. Setup | done |
| 1. Data / EDA | done |
| 2. Baseline | — |
| 3. NER | — |
| 4. BERT classification | — |
| 5. Evaluation | — |
| 6. Streamlit app | — |
| 7. README / tests | — |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python -m src.data --config configs/data.yaml   # download + clean + split both datasets
pytest
```

Data is downloaded by `src/data.py` into `data/` and is never committed.
EDA: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

## Data

### MTSamples (specialty classification)

- **Original source:** [mtsamples.com](https://www.mtsamples.com), a collection of sample medical transcription reports.
- **Copy used:** Hugging Face dataset [`harishnair04/mtsamples`](https://huggingface.co/datasets/harishnair04/mtsamples) (file `mtsamples.csv`, 4,999 rows), pinned to commit `139c3fb2d00cd390766b15ff905f88dd40e0faee`. This file has the same columns as the widely used Kaggle dataset [`tboyle10/medicaltranscriptions`](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions), which was scraped from mtsamples.com.
- **License / terms: to verify.** The Hugging Face copy is tagged `apache-2.0` by its uploader, but its dataset card is empty and states no source. The rights belong to mtsamples.com, whose terms could not be checked automatically (the site blocks scripts). The Kaggle license is still to be checked. The data is used here only for non-commercial academic work.
- Only the `transcription` column is used as model input. The `description` and `keywords` columns often name the specialty directly, so using them would leak the answer.

#### Cleaning

MTSamples files most reports twice: under the real specialty **and** under a catch-all label (Surgery, Consult, ...). We collapse these duplicates, remove truncated scrapes (fewer than 50 words), drop 10 labels that are not specialties, and keep a report only if **exactly one specialty is left and it is one of our 8 classes**.

The 50-word threshold was chosen by reading the short reports. Below 50 words almost all of them stop mid-sentence (e.g. `"INDICATIONS:,"` and nothing else). Above it, reports are short but complete (a tilt test, lung function results).

Non-specialty labels removed: Surgery, Consult - History and Phy., SOAP / Chart / Progress Notes, Discharge Summary, Emergency Room Reports, Office Notes, Letters, IME-QME-Work Comp etc., General Medicine, Radiology. (Radiology is included because imaging reports are filed under Radiology *and* the specialty of the organ imaged.)

| Step | Removed | Remaining |
|---|---:|---:|
| Raw rows | | 4,999 |
| Empty transcription | 33 | 4,966 |
| Duplicate rows (same report under several labels) | 2,609 | 2,357 |
| Too short (< 50 words, truncated scrape) | 41 | 2,316 |
| Only catch-all labels | 292 | 2,024 |
| Two or more specialties (ambiguous) | 308 | 1,716 |
| Single specialty, not one of our 8 | 434 | **1,282** |

#### Classes and split

Stratified 70/15/15 split with seed 42, saved to `data/processed/mtsamples/{train,val,test}.csv`. Every model uses these files.

| Specialty | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Cardiovascular / Pulmonary | 233 | 50 | 50 | 333 |
| Orthopedic | 153 | 33 | 33 | 219 |
| Gastroenterology | 129 | 27 | 28 | 184 |
| Obstetrics / Gynecology | 98 | 21 | 21 | 140 |
| Urology | 92 | 19 | 20 | 131 |
| Neurology | 86 | 19 | 18 | 123 |
| ENT - Otolaryngology | 55 | 12 | 12 | 79 |
| Ophthalmology | 51 | 11 | 11 | 73 |
| **Total** | **897** | **192** | **193** | **1,282** |

The test set is small: ENT and Ophthalmology have only 11–12 test reports, so their per-class scores vary a lot.

### MACCROBAT (entity extraction)

- **Source:** [MACCROBAT2020 on figshare](https://doi.org/10.6084/m9.figshare.9764942.v2). The zip's MD5 checksum is checked on download.
- **License:** CC BY 4.0. Citation: Caufield, J. Harry (2019). *MACCROBAT*. figshare. Dataset. https://doi.org/10.6084/m9.figshare.9764942.v2
- **Content:** 200 clinical case reports from PubMed Central (published articles, not patient records), annotated by hand in brat format with 41 entity types.

#### Label mapping

| Our label | MACCROBAT labels | Spans |
|---|---|---:|
| SYMPTOM | `Sign_symptom` | 3,380 |
| TREATMENT | `Medication` + `Therapeutic_procedure` | 2,113 |

Why this mapping:
- **SYMPTOM ← Sign_symptom.** This is MACCROBAT's label for what the patient experiences or shows. It also covers clinical findings ("mass", "lesion"), so our SYMPTOM really means *signs and symptoms*.
- **TREATMENT ← Medication + Therapeutic_procedure.** A treatment can be a drug ("prednisone") or a procedure done to treat the patient ("resection", "ablation"). Each label on its own would miss half of the treatments.
- **Not mapped:**
  - `Disease_disorder` (1,362 spans): a disease is a diagnosis, not a symptom. Merging it into SYMPTOM would mix two different things.
  - `Diagnostic_procedure` (4,598 spans): tests like a CT scan find a problem but don't treat it.
  - All other labels are ignored. The model therefore learns that these words are *not* SYMPTOM or TREATMENT.
- **Why not BC5CDR:** its labels are Chemical and Disease, so we would have to map diseases to symptoms, which is the error described above.

Conversion: discontinuous entities (1 among the mapped labels) are skipped, and overlapping mapped entities (4) keep the longer one. Documents are split 70/15/15 (140/30/30) with seed 42, saved to `data/processed/maccrobat/*.spacy`.

## Findings from the EDA

- **62.6% of reports are longer than BERT's 512-token limit**, and truncation cuts 40% of all tokens. BERT only sees the beginning of each report; the TF-IDF baseline sees all of it.
- **Report type may be a shortcut.** 41% of reports are operative notes, but Neurology has none (0%), against 31–63% for every other class. A model could learn "operative note → not Neurology" instead of medical content. The evaluation phase checks scores separately on operative and non-operative reports.
- **Domain shift for NER.** The NER model is trained on published case reports (MACCROBAT) and applied to dictated transcriptions (MTSamples), so it will do worse on MTSamples than on its own test set.

## Plan for the next phases

- **Phase 2 — Baseline:** TF-IDF + logistic regression with class weights; macro-F1 reported as mean ± std over 5 seeds.
- **Phase 3 — NER:** EntityRuler dictionary baseline vs a trained spaCy NER model; entity-level precision/recall/F1 per label.
- **Phase 4 — BERT:** fine-tune Bio_ClinicalBERT on Colab. Since 62.6% of reports exceed 512 tokens, compare at least two ways to handle long reports:
  1. **plain truncation:** keep the first 512 tokens;
  2. **head + tail:** keep the first and last tokens (e.g. 128 + 382), since the end of a report often holds the assessment and plan;
  3. *(if time allows)* **chunking:** split the report into 512-token windows and average the predicted probabilities.
- **Phase 5 — Evaluation:** compare every model with the baseline, show confusion matrices, and check the report-type shortcut (scores on operative vs non-operative test reports).
- **Phase 6 — Streamlit app**, **Phase 7 — README / tests.**

## Project structure

```
configs/     hyperparameters and data settings (yaml)
data/        raw/ and processed/ data (not committed)
notebooks/   01_eda.ipynb
src/         utils.py, data.py, (baseline.py, ner.py, classify.py, evaluate.py)
tests/       pytest tests
outputs/runs/ one folder per experiment: config + metrics
```
