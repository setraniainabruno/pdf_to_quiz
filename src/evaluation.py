from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .generation import Quiz


# --------------------------------------------------------------------------- #
# Bloc 2 : métriques de classification (passage quizable / non quizable)
# --------------------------------------------------------------------------- #

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Retourne [[TN, FP], [FN, TP]] pour des labels binaires 0/1."""
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray  # [[TN, FP], [FN, TP]]

    def report(self) -> str:
        tn, fp, fn, tp = self.confusion_matrix.ravel()
        return (
            f"Accuracy  : {self.accuracy:.3f}\n"
            f"Precision : {self.precision:.3f}\n"
            f"Recall    : {self.recall:.3f}\n"
            f"F1-score  : {self.f1:.3f}\n"
            f"Matrice de confusion :\n"
            f"                Prédit non-quizable   Prédit quizable\n"
            f"Réel non-quizable       {tn:>5}                {fp:>5}\n"
            f"Réel quizable           {fn:>5}                {tp:>5}\n"
        )


def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    """Calcule les métriques standard de classification binaire.

    y_true / y_pred : tableaux 0/1 (0 = non quizable, 1 = quizable).
    À utiliser sur un jeu de VALIDATION annoté manuellement, distinct du
    jeu (éventuellement heuristique) utilisé pour l'entraînement --
    sinon les métriques ne mesurent que la capacité du modèle à retrouver
    les règles heuristiques, pas sa qualité réelle.
    """
    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred),
        recall=recall_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred),
        confusion_matrix=confusion_matrix(y_true, y_pred),
    )


# --------------------------------------------------------------------------- #
# Bloc 3/4 : contrôles structurels du quiz généré
# --------------------------------------------------------------------------- #

@dataclass
class QuizQualityReport:
    n_questions: int
    n_skipped: int
    generation_success_rate: float
    n_duplicate_option_sets: int   # questions dont les 4 options ne sont pas toutes distinctes
    avg_question_length: float
    pages_covered: int

    def report(self) -> str:
        return (
            f"Questions générées         : {self.n_questions}\n"
            f"Passages ignorés            : {self.n_skipped}\n"
            f"Taux de succès génération   : {self.generation_success_rate:.1%}\n"
            f"Questions avec options dupliquées : {self.n_duplicate_option_sets}\n"
            f"Longueur moyenne de question : {self.avg_question_length:.0f} caractères\n"
            f"Pages du document couvertes  : {self.pages_covered}\n"
        )


def evaluate_quiz_quality(quiz: Quiz) -> QuizQualityReport:
    """Contrôles structurels automatiques -- ne mesurent PAS la pertinence
    pédagogique de la question (cela nécessite une évaluation humaine ou
    des questions de référence, voir docstring du module)."""
    n_total_attempts = len(quiz.questions) + quiz.skipped
    success_rate = len(quiz.questions) / n_total_attempts if n_total_attempts else 0.0

    n_duplicates = sum(
        1 for q in quiz.questions if len(set(o.strip().lower() for o in q.options)) < len(q.options)
    )

    avg_len = (
        float(np.mean([len(q.question) for q in quiz.questions])) if quiz.questions else 0.0
    )

    pages_covered = len({q.source_page for q in quiz.questions})

    return QuizQualityReport(
        n_questions=len(quiz.questions),
        n_skipped=quiz.skipped,
        generation_success_rate=success_rate,
        n_duplicate_option_sets=n_duplicates,
        avg_question_length=avg_len,
        pages_covered=pages_covered,
    )