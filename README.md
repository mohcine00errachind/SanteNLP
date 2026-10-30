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
| 2. Baseline | done |
| 3. NER | done |
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
python -m src.baseline --config configs/baseline.yaml   # add --subset 200 for a quick run
python -m src.ner --config configs/ner.yaml                              # dictionary + spaCy NER (~10 min, CPU)
python -m src.ner --config configs/ner.yaml --vectors en_core_web_md     # same, with word vectors
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

## Results

### Specialty classification

All scores are on the same fixed test set (193 reports).

| Model | Test macro-F1 (5 seeds) | Test macro-F1 (main model) | Accuracy (main) |
|---|---:|---:|---:|
| TF-IDF + logistic regression | **0.940 ± 0.003** | 0.954 | 0.948 |
| Bio_ClinicalBERT | *Phase 4* | | |

**How the seeds work.** TF-IDF + logistic regression is deterministic: on the same data, every seed gives exactly the same model (checked in `tests/test_baseline.py`). The seeds therefore vary the *data* instead. Each seed draws a new stratified train/val split from the saved train+val reports, and the test set stays fixed. The ± is the sample standard deviation over the 5 seeds. BERT will use the same 5 train/val splits.

**Baseline setup.** Word unigrams and bigrams, `sublinear_tf`, `min_df=2`, balanced class weights. C is chosen on the val split from {0.1, 1, 10, 100, 1000}; the main model picks C = 100. Config: [`configs/baseline.yaml`](configs/baseline.yaml).

**Main model per class (test):**

| Specialty | Precision | Recall | F1 | n |
|---|---:|---:|---:|---:|
| Cardiovascular / Pulmonary | 0.91 | 0.98 | 0.94 | 50 |
| ENT - Otolaryngology | 1.00 | 1.00 | 1.00 | 12 |
| Gastroenterology | 0.92 | 0.86 | 0.89 | 28 |
| Neurology | 0.86 | 1.00 | 0.92 | 18 |
| Obstetrics / Gynecology | 1.00 | 1.00 | 1.00 | 21 |
| Ophthalmology | 1.00 | 1.00 | 1.00 | 11 |
| Orthopedic | 1.00 | 0.97 | 0.98 | 33 |
| Urology | 1.00 | 0.80 | 0.89 | 20 |

**What the baseline tells us**
- **The task is easy for a bag-of-words model.** Specialties have very distinctive vocabulary (colonoscopy, carpal tunnel, fetal). This is not caused by duplicates: no test report is closer than cosine 0.86 to any training report. BERT has little room to improve, and with 193 test reports, one error is worth about 0.5 points.
- **All 10 errors are non-operative reports.** The error rate is 0% on the 86 operative test reports and 9.3% (10 / 107) on the non-operative ones (consults, follow-ups). These notes contain long general histories and get pulled toward Cardio (5 errors) or Neurology (3 errors), the one class made only of non-operative notes. This is the report-type effect seen in the EDA.

### Entity extraction (MACCROBAT)

All scores are on the MACCROBAT test split (30 documents, 656 sentences, 431 SYMPTOM and 392 TREATMENT entities). Models train on the 140 training documents, cut into 3,203 sentences.

| Model | Exact macro-F1 | Overlap macro-F1 | SYMPTOM P / R / F1 | TREATMENT P / R / F1 |
|---|---:|---:|---|---|
| Dictionary (EntityRuler, 1,832 phrases) | 0.580 | 0.634 | 0.69 / 0.61 / **0.65** | 0.72 / 0.39 / 0.51 |
| spaCy NER, no vectors (5 seeds) | 0.588 ± 0.017 | 0.652 ± 0.023 | 0.66 / 0.58 / 0.62 | 0.69 / 0.48 / 0.56 |
| spaCy NER + `en_core_web_md` vectors (5 seeds) | **0.602 ± 0.005** | **0.670 ± 0.008** | 0.68 / 0.58 / 0.62 | 0.73 / 0.49 / **0.58** |

Per-label scores are exact match, averaged over seeds. ± is the sample std over 5 seeds (initialization, shuffling and dropout change; the data does not). The dictionary is deterministic.

**Models**
- **Dictionary:** every phrase annotated in the training documents, kept if it was annotated at least `min_count` times and if, of all its occurrences in the training text, at least `min_precision` are annotated with that label. Both thresholds are chosen on val (best: 1 and 0.3). The precision filter drops words like "treatment" that are usually not annotated.
- **spaCy NER:** spaCy's default NER architecture (CNN token encoder + transition-based parser), trained from scratch with early stopping on val macro-F1. The "+ vectors" version also reads 300-dimensional pretrained word vectors from `en_core_web_md` (general English). [scispaCy](https://allenai.github.io/scispacy/)'s biomedical models would fit the domain better, but they require `numpy<2`, which conflicts with this project's stack.
- **Kept for the app:** the vectors model with the best val score (`models/ner/spacy_ner_vectors`).

**Exact vs overlap.** *Exact* counts a prediction only if start, end and label all match the annotation. *Overlap* accepts any prediction that overlaps an annotated entity of the same label: predicting "murmur" for the gold span "regurgitation murmur" is wrong under exact match but correct under overlap. Overlap F1 is 0.05–0.07 higher for every model, so part of the errors are boundary disagreements, not missed entities. Most errors are still real misses.

**What the comparison shows (honestly)**
- **Vectors vs no vectors:** +0.014 exact macro-F1 on test, which is *within* the no-vectors seed std (0.017), and the per-seed test scores overlap (0.571–0.614 vs 0.594–0.606). On test alone this is not a clear improvement. Two things are clear, though:
  - Vectors make training much more **stable** (std 0.005 vs 0.017).
  - On val, every vectors run (0.602–0.613) beats every no-vectors run (0.561–0.576).
- **Vectors vs dictionary: a small improvement, not a clear win.**
  - The vectors model scores 0.602 ± 0.005 exact macro-F1 against 0.580 (+0.022), and 0.670 ± 0.008 overlap against 0.634 (+0.036). All 5 vectors runs score above the dictionary on test (lowest: 0.594).
  - The gap is larger than the vectors model's seed std, but it is only about 1.3× the no-vectors seed std (0.017).
  - The seed std only measures training randomness, not how much scores would change on a different set of test documents. With only 30 test documents, that second source of noise is probably larger.
  - The dictionary is still better on SYMPTOM.
- **The trained models are not much better than a word list.** With 3,203 training sentences and general-English vectors, the statistical model learns little beyond the training vocabulary. A biomedical transformer (optional in Phase 4) is the natural next step.

**SYMPTOM vs TREATMENT**
- **SYMPTOM** is the dictionary's best label (F1 0.65, higher than the trained models' 0.62). Symptoms are a fairly closed vocabulary that repeats from report to report (pain, fever, dyspnea, nausea), so a list of seen phrases covers much of the test set.
- **TREATMENT** is open-ended: drug names and procedure descriptions vary a lot, and many test treatments never appear in the training documents. The dictionary finds only 39% of them (recall 0.39). The trained models generalize somewhat better (recall 0.48–0.49, F1 0.56–0.58).
- For every model, precision (about 0.7) is higher than recall (0.4–0.6): the models miss entities more often than they invent them.

**Domain shift: MTSamples examples.** There are no gold annotations for MTSamples, so this is a qualitative check. We ran the dictionary, the no-vectors model and the vectors model on the beginning of three MTSamples test reports:
- **Orthopedic operative note:**
  - All models find the shoulder pain.
  - All three miss the procedures listed under "procedure performed" (an arthroscopy, a decompression and a tendon repair). In MACCROBAT, treatments appear inside sentences, not as numbered lists of procedure names.
  - The no-vectors model tags a body location as a symptom. The vectors model avoids this but tags an anesthesia-technique word as a treatment.
- **Gastroenterology referral letter:**
  - The symptoms are found (bleeding, weight loss, nausea, vomiting), but they are reported **even when negated** (the letter says the patient has *no* nausea).
  - Both trained models tag the verb "denies" itself as a symptom.
  - No model finds the medications (a brand-name heart drug and eye drops).
- **Neurology history note:**
  - Most symptoms are found (gait difficulty, weakness, masses, lymphadenopathy).
  - The **vectors model finds 5 of the 8 drugs** in a comma-separated medication list; the dictionary and the no-vectors model find none. This is the clearest benefit of the vectors.
  - Some diseases from the past-history list (e.g. hypercholesterolemia) are tagged as symptoms.

Takeaways for the app:
- **Negation** must be handled separately (planned rule-based flag, Phase 6). MACCROBAT annotates negated symptoms as entities, so the models learn to tag them.
- **Medication and procedure lists** in transcriptions are the weakest point.
- **Disease vs symptom confusion** remains. The dictionary also learned "hypertension" as a SYMPTOM, which means MACCROBAT annotators sometimes labeled it `Sign_symptom`.

## Plan for the next phases

- **Phase 4 — BERT:** fine-tune Bio_ClinicalBERT on Colab. Since 62.6% of reports exceed 512 tokens, compare at least two ways to handle long reports:
  1. **plain truncation:** keep the first 512 tokens;
  2. **head + tail:** keep the first and last tokens (e.g. 128 + 382), since the end of a report often holds the assessment and plan;
  3. *(if time allows)* **chunking:** split the report into 512-token windows and average the predicted probabilities.

  Besides overall macro-F1, compare BERT and the baseline **on the 107 non-operative test reports**, where all 10 baseline errors are, and list which of those 10 errors BERT fixes or repeats.

  *Optional:* fine-tune Bio_ClinicalBERT for **token classification** (SYMPTOM / TREATMENT) on MACCROBAT on Colab, with the same document split, and compare it with the spaCy models and the dictionary.
- **Phase 5 — Evaluation:** compare every model with the baseline, show confusion matrices, and check the report-type shortcut (scores on operative vs non-operative test reports). If BERT does not clearly beat the baseline, say so plainly, with the reason (distinctive vocabulary, small test set). A difference smaller than the seed std is not an improvement.
- **Phase 6 — Streamlit app:** paste a report, see highlighted entities and the predicted specialty. Includes a **rule-based negation flag**: an entity is marked "negated" when it follows a cue such as "no", "denies" or "negative for" in the same clause. The NER models tag "nausea" in "no nausea" because MACCROBAT annotates negated symptoms too.
- **Phase 7 — README / tests.**

## Project structure

```
configs/     hyperparameters and data settings (yaml)
data/        raw/ and processed/ data (not committed)
notebooks/   01_eda.ipynb
src/         utils.py, data.py, baseline.py, evaluate.py, ner.py, (classify.py)
tests/       pytest tests
outputs/runs/ one folder per experiment: config + metrics
```
