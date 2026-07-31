from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification import QuizableClassifier  # noqa: E402
from src.dataset import (  # noqa: E402
    corpus_statistics,
    load_annotations,
    to_training_arrays,
    train_val_split,
)
from src.evaluation import evaluate_classifier  # noqa: E402
from src.utils import CONFIG  # noqa: E402

MIN_RECOMMENDED = 40


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîne et évalue le classifieur quizable/non quizable.")
    parser.add_argument("--dataset", default="data/annotations.jsonl", help="Chemin du fichier d'annotations")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Proportion réservée à la validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    annotations = load_annotations(args.dataset)
    stats = corpus_statistics(annotations)

    print("=== Statistiques du corpus annoté ===")
    print(f"Passages annotés      : {stats['n_passages']}")
    print(f"Documents distincts    : {stats['n_documents']}")
    print(f"Quizables / non        : {stats['n_quizable']} / {stats['n_non_quizable']}")
    print(f"Longueur moyenne       : {stats['avg_passage_length']:.0f} caractères")
    print(f"Part annotée à la main : {stats['pct_manuel']:.0%}\n")

    if stats["n_passages"] < MIN_RECOMMENDED:
        print(
            f"⚠️  Moins de {MIN_RECOMMENDED} passages annotés : le classifieur risque de mal généraliser.\n"
            f"    Utilise `streamlit run app/annotate.py` pour en annoter davantage avant d'aller plus loin.\n"
        )
        if stats["n_passages"] < 10:
            print("Trop peu de données pour un entraînement significatif. Arrêt.")
            return

    train_set, val_set = train_val_split(annotations, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Split : {len(train_set)} exemples d'entraînement, {len(val_set)} de validation.\n")

    train_texts, train_labels = to_training_arrays(train_set)
    val_texts, val_labels = to_training_arrays(val_set)

    CONFIG.check()  # vérifie la clé API OpenAI (nécessaire pour les embeddings)

    print("Entraînement du classifieur (embeddings OpenAI + régression logistique)...")
    classifier = QuizableClassifier()
    classifier.fit(train_texts, train_labels)

    if val_set:
        val_scores = classifier.predict_proba(val_texts)
        val_preds = (val_scores >= 0.5).astype(int)
        metrics = evaluate_classifier(val_labels, val_preds)
        print("\n=== Métriques sur le jeu de VALIDATION ===")
        print(metrics.report())
    else:
        print("\n⚠️  Pas d'exemples de validation (corpus trop petit) : métriques non calculées.")

    classifier.save()
    print(f"Modèle sauvegardé dans {CONFIG.classifier_path}")
    print("L'app Streamlit principale le chargera automatiquement au prochain lancement.")


if __name__ == "__main__":
    main()
