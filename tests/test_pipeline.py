import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import clean_text, split_into_passages
from src.classification import heuristic_quizable_score
from src.evaluation import evaluate_classifier
from src.dataset import Annotation, corpus_statistics, train_val_split


# --------------------------------------------------------------------------- #
# utils.py
# --------------------------------------------------------------------------- #

def test_clean_text_recolle_mots_coupes():
    assert clean_text("informa-\ntion") == "information"


def test_clean_text_compacte_espaces():
    assert clean_text("Bonjour   le    monde") == "Bonjour le monde"


def test_clean_text_preserve_paragraphes():
    result = clean_text("Paragraphe 1.\n\nParagraphe 2.")
    assert "\n\n" in result


def test_split_into_passages_fusionne_les_courts():
    text = "Court.\n\nEncore un bout court.\n\n" + ("Long paragraphe. " * 30)
    passages = split_into_passages(text, min_chars=50, max_chars=2000)
    assert all(len(p) >= 25 for p in passages)  # aucun fragment dérisoire isolé


def test_split_into_passages_decoupe_les_longs():
    long_text = "Phrase numéro un. " * 200
    passages = split_into_passages(long_text, min_chars=10, max_chars=300)
    assert all(len(p) <= 320 for p in passages)  # marge pour la dernière phrase du bloc


# --------------------------------------------------------------------------- #
# classification.py (heuristique, sans dépendance à sentence-transformers)
# --------------------------------------------------------------------------- #

def test_heuristic_favorise_passage_factuel():
    factuel = "Paris est la capitale de la France depuis 508. Elle compte 2 millions d'habitants."
    vide = "et donc voilà."
    assert heuristic_quizable_score(factuel) > heuristic_quizable_score(vide)


def test_heuristic_score_dans_les_bornes():
    for text in ["", "a", "Un texte normal avec quelques mots.", "1234567890 " * 50]:
        score = heuristic_quizable_score(text)
        assert 0.0 <= score <= 1.0


# --------------------------------------------------------------------------- #
# evaluation.py
# --------------------------------------------------------------------------- #

def test_evaluate_classifier_cas_parfait():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])
    metrics = evaluate_classifier(y_true, y_pred)
    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


def test_evaluate_classifier_confusion_matrix_shape():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 1, 0])
    metrics = evaluate_classifier(y_true, y_pred)
    assert metrics.confusion_matrix.shape == (2, 2)


# --------------------------------------------------------------------------- #
# dataset.py
# --------------------------------------------------------------------------- #

def _make_annotations(n_pos: int, n_neg: int) -> list[Annotation]:
    annotations = []
    for i in range(n_pos):
        annotations.append(Annotation(f"pos{i}", f"texte quizable numéro {i}", "doc.pdf", 1, label=1))
    for i in range(n_neg):
        annotations.append(Annotation(f"neg{i}", f"texte vide {i}", "doc.pdf", 1, label=0))
    return annotations


def test_corpus_statistics_compte_correctement():
    annotations = _make_annotations(n_pos=3, n_neg=2)
    stats = corpus_statistics(annotations)
    assert stats["n_passages"] == 5
    assert stats["n_quizable"] == 3
    assert stats["n_non_quizable"] == 2
    assert stats["n_documents"] == 1


def test_corpus_statistics_vide():
    stats = corpus_statistics([])
    assert stats["n_passages"] == 0


def test_train_val_split_preserve_toutes_les_annotations():
    annotations = _make_annotations(n_pos=10, n_neg=10)
    train, val = train_val_split(annotations, val_ratio=0.2, seed=0)
    assert len(train) + len(val) == len(annotations)
    assert {a.passage_id for a in train + val} == {a.passage_id for a in annotations}


def test_train_val_split_est_stratifie():
    annotations = _make_annotations(n_pos=20, n_neg=20)
    train, val = train_val_split(annotations, val_ratio=0.25, seed=0)
    val_pos = sum(1 for a in val if a.label == 1)
    val_neg = sum(1 for a in val if a.label == 0)
    # Avec un ratio 50/50 au départ, le split stratifié doit rester équilibré des deux côtés
    assert abs(val_pos - val_neg) <= 1