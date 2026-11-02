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
| 4. BERT classification | done |
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
python -m src.classify --smoke                  # BERT: tiny CPU check (full training: Colab, below)
python -m src.compare --config configs/compare.yaml   # BERT vs baseline (needs the BERT runs and models)
pytest
```

### Training BERT on Colab

Fine-tuning needs a GPU, so it runs on Google Colab with [`notebooks/colab_train_bert.ipynb`](notebooks/colab_train_bert.ipynb):

1. In Colab: *File → Open notebook → GitHub*, paste this repo's URL and open `notebooks/colab_train_bert.ipynb`.
2. *Runtime → Change runtime type → T4 GPU*, then run the cells in order. The notebook mounts Google Drive, clones the repo, installs the pinned requirements (except torch, which Colab provides), rebuilds the data, checks that the split is identical to the local one, and trains. Each training cell stops at once if Drive is not mounted, and lists the files it saved on Drive. After a disconnect, the notebook's first cell says which steps to rerun.
3. Results are saved to `MyDrive/SanteNLP/`. Copy them into the project; with Google Drive for desktop:
   ```powershell
   Copy-Item -Recurse "G:\My Drive\SanteNLP\outputs\runs\*" outputs\runs\
   New-Item -ItemType Directory -Force models\bert_specialty
   Copy-Item -Recurse "G:\My Drive\SanteNLP\models\bert_specialty\*" models\bert_specialty\
   ```
   The models must end up in `models/bert_specialty/head` and `models/bert_specialty/head_tail`.

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
| TF-IDF + logistic regression | 0.940 ± 0.003 | 0.954 | 0.948 |
| Bio_ClinicalBERT, `head` (first 510 tokens) | **0.968 ± 0.002** | **0.961** | **0.959** |
| Bio_ClinicalBERT, `head_tail` (first 128 + last 382 tokens) | 0.954 ± 0.008 | 0.954 | 0.953 |

**How the seeds work.** TF-IDF + logistic regression is deterministic: on the same data, every seed gives exactly the same model (checked in `tests/test_baseline.py`). The seeds therefore vary the *data* instead. Each seed draws a new stratified train/val split from the saved train+val reports, and the test set stays fixed. The ± is the sample standard deviation over the 5 seeds. BERT uses the same 5 train/val splits, so for each seed BERT and the baseline are trained on exactly the same reports.

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

### BERT vs baseline (Phase 4)

**Setup.** [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) (pinned revision) with a classification layer, fine-tuned for 4 epochs (learning rate 2e-5, batch 16, fp16) on a Colab T4 GPU. The epoch with the best val macro-F1 is kept. Config: [`configs/classify.yaml`](configs/classify.yaml). For long reports, `head` keeps the first 510 tokens; `head_tail` keeps the first 128 and the last 382. The comparison below is made by [`src/compare.py`](src/compare.py) from the saved test predictions (config: [`configs/compare.yaml`](configs/compare.yaml)).

**Reproducibility check.** BERT was trained twice on Colab. The first run's files were lost (a notebook bug wrote them to the Colab machine instead of Google Drive), but its scores were in the Colab log. Same code, data and seeds, retrained from scratch:

| | First run (log only) | Second run (saved, used here) |
|---|---:|---:|
| `head`, 5 seeds | 0.968 ± 0.004 | 0.968 ± 0.002 |
| `head`, main model | 0.961 | 0.961 |
| `head_tail`, 5 seeds | 0.953 ± 0.010 | 0.954 ± 0.008 |

GPU training is not bit-for-bit deterministic (fp16, GPU kernels), so small differences are expected. They are well within the seed std.

**Overall.** `head` beats the baseline in every seed: 6–7 test errors against 12–13 for the baseline on the same split. That is +0.028 macro-F1 on average (0.968 vs 0.940), about 10× the seed std. Three caveats:
- **On the main split the gap is small:** 0.961 vs 0.954 (8 vs 10 errors). This split happens to suit the baseline: its main score (0.954) is above all 5 of its seed scores.
- **No single comparison is statistically significant.** Only reports where exactly one model is right count in a McNemar test. On the main models that is 6 against 4 reports (p = 0.75); per seed, p is 0.11–0.23. The seed std only measures training randomness, not the choice of the 193 test reports. A gain of about 6 reports is consistent across seeds, but too small to be certain with this test set.
- `head_tail` (0.954 ± 0.008) is not clearly better than the baseline.

**By report type.** As planned, the comparison is also made on the 107 non-operative test reports, where all 10 baseline errors are. Macro-F1 on a subset is averaged over the classes present in it (there is no operative Neurology report).

| Test reports | Baseline (5 seeds) | `head` (5 seeds) | `head_tail` (5 seeds) |
|---|---:|---:|---:|
| Non-operative (107) | 0.904 ± 0.014, 10–13 errors | **0.949 ± 0.006**, 5–7 errors | 0.928 ± 0.009, 7–10 errors |
| Operative (86) | 0.991 ± 0.010, 0–2 errors | 0.996 ± 0.006, 0–1 errors | 0.998 ± 0.005, 0–1 errors |
| Longer than 510 tokens (127) | 0.923 ± 0.009 | **0.967 ± 0.005** | 0.948 ± 0.012 |
| Up to 510 tokens (66) | 0.971 ± 0.006 | 0.978 ± 0.004 | 0.965 ± 0.015 |

Almost all of BERT's gain is on non-operative reports, where the baseline was weak. Operative reports were already nearly perfect for every model. BERT `head` also gains most on long reports, even though it reads only their first 510 tokens while the baseline reads everything.

**The 10 baseline errors** (main models; "seeds" = how many of the 5 seed models get the report right):

| Test row | True specialty | Baseline | `head` (seeds right) | `head_tail` (seeds right) |
|---:|---|---|---|---|
| 26 | Urology | Cardio | ✅ (5/5) | ✅ (5/5) |
| 67 | Gastroenterology | Cardio | ✅ (4/5) | ✅ (0/5) |
| 80 | Orthopedic | Neurology | ✅ (5/5) | ✅ (5/5) |
| 136 | Urology | Cardio | ✅ (5/5) | ❌ Neurology (0/5) |
| 143 | Gastroenterology | Cardio | ✅ (5/5) | ✅ (5/5) |
| 191 | Gastroenterology | Neurology | ✅ (5/5) | ✅ (5/5) |
| 125 | Gastroenterology | Cardio | ❌ Cardio (3/5) | ❌ Cardio (5/5) |
| 52 | Urology | Gastroenterology | ❌ Gastro (0/5) | ❌ Gastro (1/5) |
| 187 | Urology | Gastroenterology | ❌ Gastro (1/5) | ❌ Gastro (0/5) |
| 68 | Cardio / Pulmonary | Neurology | ❌ Neurology (0/5) | ❌ Neurology (0/5) |

(Row 125: the main models are wrong, but 3 of 5 `head` seed models and all 5 `head_tail` seed models get it right. Row 67: the `head_tail` main model is right, but none of its seed models are.)

- **BERT `head` fixes 6 and repeats 4.** Five of the fixes hold in all 5 seeds. The baseline's typical error, pulling general non-operative notes toward Cardio, mostly disappears: across the 5 seeds, 29 of the baseline's 61 wrong predictions are Cardio, against 3 of BERT `head`'s 32.
- **The repeated errors look like hard or ambiguous reports, not model failures.** Rows 52 and 187 are Urology reports seen in an acute setting (flank pain, blood in the urine) that read much like abdominal complaints. Row 68 is labelled Cardio / Pulmonary but describes a neurological emergency (an unresponsive patient with a severe headache); every model and every seed says Neurology.
- **BERT makes 4 new errors** (reports the baseline gets right), and all 4 are predicted **Gastroenterology**: two Urology reports (kidney stone, follow-up of a circumcision), an Ob/Gyn operation that also removed an abdominal wall mass, and an Orthopedic report that starts with a long family and social history. Across seeds, 19 of BERT `head`'s 32 wrong predictions are Gastroenterology. BERT seems to read "abdomen / flank" as Gastroenterology even when the report belongs to a neighbouring specialty. The baseline's Cardio bias is replaced by a smaller Gastroenterology bias.

**Why `head_tail` is worse than `head`.** The idea was that the end of a report holds the assessment and plan. The data says otherwise:
- **The beginning of a report carries more specialty information than the end.** We retrained the TF-IDF baseline on only the tokens each strategy keeps (same tokenizer, same 5 seeds). On exactly the text BERT sees, the baseline drops from 0.928 ± 0.008 (`head` window) to 0.914 ± 0.010 (`head_tail` window), and from 0.921 to 0.886 on long reports. At equal length, the first 382 tokens beat the last 382 (0.919 vs 0.904). This test doesn't involve BERT at all, so it isn't training noise.
- **What is at the beginning and at the end.** In the 549 long training reports, the first 510 tokens hold the headers that name the problem: PREOPERATIVE DIAGNOSIS, PROCEDURE PERFORMED, HISTORY OF PRESENT ILLNESS. The last 382 tokens hold mostly the body-system exam (HEENT, LUNGS, ABDOMEN, EXTREMITIES, NEUROLOGIC), which looks the same in every specialty. IMPRESSION / PLAN is in the tail of only 70 of the 549 (13%). To keep this generic text, `head_tail` drops the middle of the report (everything after token 128 that is not in the last 382), which is often the procedure description or the history.
- **The general exam may also pull toward Neurology.** Across seeds, 15 of `head_tail`'s wrong predictions are Neurology, against 5 for `head`. This fits the report-type shortcut from the EDA (Neurology is made only of non-operative notes, which end with a general exam), but it is an interpretation, not a test.
- **The gap is small but consistent.** `head` has fewer errors than `head_tail` on every seed's split (the same split for both), on average 2.4 reports. About a third of that gap is on reports of 510 tokens or fewer, where both strategies see exactly the same text. So part of the gap is training noise, not the shortening. The main models differ on only 7 test reports (4 vs 3).
- `head_tail` does help on single reports. Row 132 is an Orthopedic report whose opening is a family and social history: `head` gets it wrong in all 5 seeds, `head_tail` gets it right in all 5.

**Kept for the app:** BERT `head` (`models/bert_specialty/head`).

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

- **Phase 4 — BERT:** done (results above). Not done: **chunking** (split a report into 512-token windows and average the probabilities) and the *optional* Bio_ClinicalBERT token classifier for NER. Since `head` already beats `head_tail` and the beginning of a report holds most of the specialty information, chunking is unlikely to change much.
- **Phase 5 — Evaluation:** confusion matrices for every model, and an estimate of the uncertainty that comes from the small test set (e.g. a bootstrap over test reports), since the seed std does not include it. Check the Gastroenterology bias of BERT on the confusions between Urology, Ob/Gyn and Gastroenterology. A difference smaller than the seed std is not an improvement.
- **Phase 6 — Streamlit app:** paste a report, see highlighted entities and the predicted specialty. Includes a **rule-based negation flag**: an entity is marked "negated" when it follows a cue such as "no", "denies" or "negative for" in the same clause. The NER models tag "nausea" in "no nausea" because MACCROBAT annotates negated symptoms too.
- **Phase 7 — README / tests.**

## Project structure

```
configs/     hyperparameters and data settings (yaml)
data/        raw/ and processed/ data (not committed)
notebooks/   01_eda.ipynb, colab_train_bert.ipynb
src/         utils.py, data.py, baseline.py, evaluate.py, ner.py, classify.py, compare.py, colab.py
tests/       pytest tests
outputs/runs/ one folder per experiment: config + metrics
```

## License

Code: [MIT](LICENSE) © 2026 Mohcine Errachid. The datasets are not part of this repository and keep their own terms (MACCROBAT: CC BY 4.0; MTSamples: see the Data section).
