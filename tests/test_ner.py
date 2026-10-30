import spacy
from spacy.tokens import Span

from src.ner import (
    build_dictionary,
    evaluate_ner,
    make_dictionary_nlp,
    phrase_statistics,
    split_into_sentences,
    to_examples,
    train_ner,
)

LABELS = ["SYMPTOM", "TREATMENT"]
nlp = spacy.blank("en")


def gold(text: str, ents: list[tuple[str, str]]):
    """Build an annotated Doc; ents = [(entity text, label)], found by text search."""
    doc = nlp.make_doc(text)
    spans = []
    for surface, label in ents:
        start = text.index(surface)
        spans.append(doc.char_span(start, start + len(surface), label=label))
    doc.ents = spans
    return doc


def test_split_into_sentences_keeps_entities():
    doc = gold("He had fever.\nHe got aspirin.", [("fever", "SYMPTOM"), ("aspirin", "TREATMENT")])
    sentences = split_into_sentences([doc])
    assert [s.text for s in sentences] == ["He had fever.", "He got aspirin."]
    assert [(e.text, e.label_) for s in sentences for e in s.ents] == [("fever", "SYMPTOM"), ("aspirin", "TREATMENT")]


def test_dictionary_drops_ambiguous_phrases():
    docs = [
        gold("fever and treatment", [("fever", "SYMPTOM"), ("treatment", "TREATMENT")]),
        gold("fever again", [("fever", "SYMPTOM")]),
        gold("treatment was delayed", []),  # "treatment" appears but is not annotated
        gold("no treatment needed", []),
    ]
    stats = phrase_statistics(docs)
    assert stats["fever"]["occurrences"] == 2 and stats["treatment"]["occurrences"] == 3

    patterns = build_dictionary(stats, min_count=1, min_precision=0.5)
    # fever: 2/2 = 1.0 -> kept ; treatment: 1/3 = 0.33 -> dropped
    assert patterns == [{"label": "SYMPTOM", "pattern": "fever"}]


def test_dictionary_ner_is_case_insensitive_and_prefers_longer_matches():
    patterns = [{"label": "SYMPTOM", "pattern": "pain"}, {"label": "SYMPTOM", "pattern": "chest pain"}]
    doc = make_dictionary_nlp(patterns)("Severe Chest Pain today")
    assert [(e.text, e.label_) for e in doc.ents] == [("Chest Pain", "SYMPTOM")]


def test_evaluate_ner_exact_match():
    golds = [gold("fever and aspirin", [("fever", "SYMPTOM"), ("aspirin", "TREATMENT")])]
    perfect = make_dictionary_nlp([{"label": "SYMPTOM", "pattern": "fever"}, {"label": "TREATMENT", "pattern": "aspirin"}])
    assert evaluate_ner(perfect, golds, LABELS)["macro_f1"] == 1.0

    half = make_dictionary_nlp([{"label": "SYMPTOM", "pattern": "fever"}])  # misses aspirin
    m = evaluate_ner(half, golds, LABELS)
    assert m["per_label"]["SYMPTOM"]["f"] == 1.0 and m["per_label"]["TREATMENT"]["f"] == 0.0
    assert m["macro_f1"] == 0.5


def test_overlap_match_counts_boundary_errors_as_correct():
    golds = [gold("regurgitation murmur and fever", [("regurgitation murmur", "SYMPTOM"), ("fever", "SYMPTOM")])]
    model = make_dictionary_nlp([{"label": "SYMPTOM", "pattern": "murmur"}])  # right entity, wrong boundary
    m = evaluate_ner(model, golds, LABELS)
    assert m["exact"]["per_label"]["SYMPTOM"] == {"p": 0.0, "r": 0.0, "f": 0.0}
    # overlap: the 1 prediction is correct (P=1), it finds 1 of 2 gold entities (R=0.5)
    assert m["overlap"]["per_label"]["SYMPTOM"] == {"p": 1.0, "r": 0.5, "f": 0.6667}


def test_exact_scores_match_spacy_scorer():
    golds = [
        gold("fever and aspirin and cough", [("fever", "SYMPTOM"), ("aspirin", "TREATMENT"), ("cough", "SYMPTOM")]),
        gold("aspirin for pain", [("pain", "SYMPTOM")]),
    ]
    model = make_dictionary_nlp([{"label": "SYMPTOM", "pattern": "fever"}, {"label": "TREATMENT", "pattern": "aspirin"}])
    ours = evaluate_ner(model, golds, LABELS)["exact"]
    spacy_scores = model.evaluate(to_examples(model, golds))
    assert ours["micro"]["f"] == round(spacy_scores["ents_f"], 4)
    for label in LABELS:
        assert ours["per_label"][label]["f"] == round(spacy_scores["ents_per_type"][label]["f"], 4)


def test_train_ner_runs_and_learns_a_tiny_task():
    docs = [gold(f"patient {i} had fever and took aspirin", [("fever", "SYMPTOM"), ("aspirin", "TREATMENT")]) for i in range(20)]
    cfg = {"labels": LABELS, "training": {"max_epochs": 15, "patience": 5, "dropout": 0.0, "batch_size": 4}}
    model, info = train_ner(docs, docs, cfg, seed=0)
    assert set(model.get_pipe("ner").labels) == set(LABELS)
    assert info["val_macro_f1"] > 0.9


def test_train_ner_with_vectors_uses_them():
    docs = [gold(f"patient {i} had fever and took aspirin", [("fever", "SYMPTOM"), ("aspirin", "TREATMENT")]) for i in range(8)]
    cfg = {"labels": LABELS, "vectors": "en_core_web_md",
           "training": {"max_epochs": 2, "patience": 5, "dropout": 0.0, "batch_size": 4}}
    model, _ = train_ner(docs, docs, cfg, seed=0)
    assert model.vocab.vectors.shape[1] == 300
    assert model.get_pipe_config("ner")["model"]["tok2vec"]["pretrained_vectors"] is True
