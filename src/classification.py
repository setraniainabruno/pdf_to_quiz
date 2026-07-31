from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from openai import OpenAI

from .extraction import Passage
from .utils import CONFIG

# Modèle d'embeddings OpenAI : rapide, pas de dépendance locale à torch,
# et gère nativement le français (contrairement à beaucoup de modèles anglais uniquement).
_EMBEDDING_MODEL_NAME = "text-embedding-3-small"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        CONFIG.check()
        _client = OpenAI(api_key=CONFIG.openai_api_key)
    return _client


def embed_passages(texts: list[str]) -> np.ndarray:
    """Calcule les embeddings d'une liste de textes via l'API OpenAI.

    Traite par lots de 100 (limite raisonnable côté API) pour rester robuste
    même si beaucoup de passages sont extraits d'un document volumineux.
    """
    client = _get_client()
    all_embeddings: list[list[float]] = []

    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=_EMBEDDING_MODEL_NAME, input=batch)
        all_embeddings.extend(item.embedding for item in response.data)

    return np.array(all_embeddings)


# --------------------------------------------------------------------------- #
# Régression logistique "from scratch" (numpy pur, sans scikit-learn/scipy)
# --------------------------------------------------------------------------- #

class SimpleLogisticRegression:
    """Régression logistique binaire entraînée par descente de gradient.

    Implémentation volontairement simple (pas de solveur optimisé type
    L-BFGS) : suffisante pour ce projet (quelques dizaines/centaines
    d'exemples, embeddings déjà normalisés par l'API), et évite toute
    dépendance à scipy/scikit-learn.
    """

    def __init__(self, lr: float = 0.5, epochs: int = 300, l2: float = 1e-3) -> None:
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Pondération "balanced" (équivalent de class_weight="balanced" de sklearn) :
        # compense un déséquilibre entre passages quizables et non quizables.
        counts = np.bincount(y, minlength=2).astype(float)
        counts[counts == 0] = 1  # évite une division par zéro si une classe est absente
        class_weights = n_samples / (2.0 * counts)
        sample_weights = class_weights[y]

        for _ in range(self.epochs):
            z = X @ self.weights + self.bias
            predictions = _sigmoid(z)
            error = (predictions - y) * sample_weights

            grad_w = (X.T @ error) / n_samples + self.l2 * self.weights
            grad_b = float(np.sum(error) / n_samples)

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Le modèle n'est pas entraîné.")
        return _sigmoid(X @ self.weights + self.bias)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # np.clip évite un overflow numérique pour de grandes valeurs négatives/positives
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# --------------------------------------------------------------------------- #
# Labelling heuristique (amorçage du jeu d'entraînement)
# --------------------------------------------------------------------------- #

_DATE_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_NUMBER_RE = re.compile(r"\b\d+([.,]\d+)?\s?(%|km|kg|m|ans?)?\b")
_DEFINITION_MARKERS = (" est un ", " est une ", " se définit comme ", " désigne ",
                        " correspond à ", " permet de ", " signifie ")
_PROPER_NOUN_RE = re.compile(r"\b[A-ZÀ-Ý][a-zà-ÿ]{2,}\b")


def heuristic_quizable_score(text: str) -> float:
    """Score heuristique [0, 1] indiquant si un passage semble contenir
    assez d'information factuelle pour générer une bonne question.

    Règles utilisées (documentées pour le rapport) :
    - présence d'une date -> +
    - présence de chiffres/nombres -> +
    - présence d'un marqueur de définition ("est un", "désigne", ...) -> +
    - présence de noms propres (entités potentielles) -> +
    - passage trop court -> -
    """
    score = 0.0
    n_signals = 0

    if _DATE_RE.search(text):
        score += 1
        n_signals += 1
    n_signals += 1  # nombre : toujours compté, même si absent
    if len(_NUMBER_RE.findall(text)) >= 1:
        score += 1

    if any(marker in text.lower() for marker in _DEFINITION_MARKERS):
        score += 1
    n_signals += 1

    proper_nouns = len(set(_PROPER_NOUN_RE.findall(text)))
    if proper_nouns >= 2:
        score += 1
    n_signals += 1

    if len(text) < 150:
        score -= 1

    return max(0.0, min(1.0, score / n_signals))


def bootstrap_labels(passages: list[Passage], threshold: float = 0.5) -> np.ndarray:
    """Génère des pseudo-labels 0/1 à partir de l'heuristique ci-dessus."""
    return np.array([1 if heuristic_quizable_score(p.text) >= threshold else 0 for p in passages])


# --------------------------------------------------------------------------- #
# Classifieur supervisé
# --------------------------------------------------------------------------- #

@dataclass
class ClassificationResult:
    passage: Passage
    is_quizable: bool
    score: float


class QuizableClassifier:
    """Classifieur binaire "passage quizable / non quizable" basé sur
    des embeddings de phrases (API OpenAI) + régression logistique maison."""

    def __init__(self) -> None:
        self.model: SimpleLogisticRegression | None = None

    def fit(self, texts: list[str], labels: np.ndarray) -> None:
        X = embed_passages(texts)
        self.model = SimpleLogisticRegression()
        self.model.fit(X, labels)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Le classifieur n'est pas entraîné. Appelle fit() ou load().")
        X = embed_passages(texts)
        return self.model.predict_proba(X)

    def save(self, path: str = CONFIG.classifier_path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str = CONFIG.classifier_path) -> None:
        self.model = joblib.load(path)

    @property
    def is_trained(self) -> bool:
        return self.model is not None


def select_quizable_passages(
    passages: list[Passage],
    classifier: QuizableClassifier | None = None,
    threshold: float = 0.5,
    top_k: int | None = None,
) -> list[ClassificationResult]:
    """Filtre les passages "quizables" parmi une liste de passages extraits.

    Si aucun classifieur entraîné n'est fourni (ou disponible sur disque),
    on retombe sur le score heuristique -- ce qui permet à l'app de
    fonctionner dès le premier lancement, sans jeu de données annoté.
    """
    if classifier is not None and classifier.is_trained:
        scores = classifier.predict_proba([p.text for p in passages])
    else:
        scores = np.array([heuristic_quizable_score(p.text) for p in passages])

    results = [
        ClassificationResult(passage=p, is_quizable=bool(s >= threshold), score=float(s))
        for p, s in zip(passages, scores)
    ]
    results.sort(key=lambda r: r.score, reverse=True)

    quizable = [r for r in results if r.is_quizable]
    if top_k is not None:
        quizable = quizable[:top_k]

    return quizable


if __name__ == "__main__":
    # Petit test manuel : python -m src.classification
    demo_passages = [
        Passage(id="p1", text="Paris est la capitale de la France depuis 508. Elle compte plus de 2 millions d'habitants.", page=1),
        Passage(id="p2", text="et donc voilà.", page=1),
        Passage(id="p3", text="Le Deep Learning est une branche du Machine Learning qui utilise des réseaux de neurones à plusieurs couches, notamment pour la reconnaissance d'image depuis 2012.", page=2),
    ]
    for r in select_quizable_passages(demo_passages, threshold=0.4):
        print(f"[{r.score:.2f}] {r.passage.text[:80]}")