import pandas as pd
import pytest
from transformers import BertConfig, BertModel, BertTokenizerFast

from src.classify import encode, load_classifier, predict_proba, save_classifier, select_tokens, train_and_evaluate

WORDS = ["heart", "murmur", "chest", "pain", "knee", "fracture", "bone", "cast", "the", "patient", "has"]


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """A tiny random BERT + tokenizer saved locally: tests run offline in seconds."""
    path = tmp_path_factory.mktemp("tiny_bert")
    vocab = {token: i for i, token in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS)}
    tokenizer = BertTokenizerFast(vocab=vocab)  # transformers 5: "vocab", not "vocab_file"
    assert tokenizer.tokenize("heart murmur") == ["heart", "murmur"]
    config = BertConfig(vocab_size=len(WORDS) + 5, hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32, max_position_embeddings=64)
    BertModel(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def test_select_tokens_head_and_head_tail():
    ids = list(range(10))
    assert select_tokens(ids, "head", max_tokens=4, head_tokens=1) == [0, 1, 2, 3]
    assert select_tokens(ids, "head_tail", max_tokens=4, head_tokens=1) == [0, 7, 8, 9]
    assert select_tokens(ids[:3], "head_tail", max_tokens=4, head_tokens=1) == [0, 1, 2]  # short: unchanged
    with pytest.raises(ValueError):
        select_tokens(ids, "middle", max_tokens=4, head_tokens=1)


def test_encode_adds_special_tokens_and_respects_max_length(tiny_model_dir):
    tokenizer = BertTokenizerFast.from_pretrained(tiny_model_dir)
    text = " ".join(WORDS)  # 11 tokens
    [ids] = encode(tokenizer, [text], "head_tail", max_length=8, head_tokens=2)
    assert len(ids) == 8
    assert ids[0] == tokenizer.cls_token_id and ids[-1] == tokenizer.sep_token_id
    # 8 - 2 special tokens = 6 text tokens: the first 2 and the last 4
    assert tokenizer.convert_ids_to_tokens(ids[1:-1]) == ["heart", "murmur", "cast", "the", "patient", "has"]


def test_train_save_reload_predict(tiny_model_dir, tmp_path):
    rows = [("the patient has heart murmur chest pain", "cardio"), ("the patient has knee fracture bone cast", "ortho")] * 8
    df = pd.DataFrame(rows, columns=["text", "label"])
    cfg = {
        "model_name": str(tiny_model_dir), "model_revision": None, "max_length": 16, "head_tokens": 4,
        "training": {"epochs": 2, "learning_rate": 1e-3, "batch_size": 4, "eval_batch_size": 8,
                     "weight_decay": 0.0, "warmup_ratio": 0.0, "fp16": False},
    }
    trainer, val_f1, test_pred, history = train_and_evaluate(
        df, df, df, ["cardio", "ortho"], cfg, "head", seed=0, checkpoint_dir=str(tmp_path / "ckpt"))
    assert len(test_pred) == len(df) and set(test_pred) <= {"cardio", "ortho"}
    assert [h["epoch"] for h in history] == [1, 2]
    assert 0.0 <= val_f1 <= 1.0

    save_classifier(trainer, tmp_path / "model", cfg, "head")
    tokenizer, model, settings = load_classifier(tmp_path / "model")
    assert settings == {"strategy": "head", "max_length": 16, "head_tokens": 4}
    probs = predict_proba(["heart murmur", "knee fracture", "pain"], tokenizer, model, settings, batch_size=2)
    assert list(probs.columns) == ["cardio", "ortho"] and len(probs) == 3
    assert probs.sum(axis=1).round(5).eq(1.0).all()
