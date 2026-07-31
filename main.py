from __future__ import annotations

import utils.clients
import utils.http_session
import utils.transport
import utils.session
import argparse
import json

from src.classification import QuizableClassifier, select_quizable_passages
from src.evaluation import evaluate_quiz_quality
from src.extraction import extract_passages
from src.generation import generate_quiz
from src.utils import CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description="Génère un quiz à partir d'un PDF.")
    parser.add_argument("pdf_path", help="Chemin vers le fichier PDF source")
    parser.add_argument("--n-questions", type=int, default=5, help="Nombre de questions à générer")
    parser.add_argument("--threshold", type=float, default=0.5, help="Seuil de sélection des passages")
    parser.add_argument("--output", default=None, help="Chemin de sortie JSON (optionnel)")
    args = parser.parse_args()

    CONFIG.check()

    print(f"[1/3] Extraction de {args.pdf_path} ...")
    passages = extract_passages(args.pdf_path)
    print(f"      -> {len(passages)} passage(s) extrait(s)")

    print("[2/3] Sélection des passages quizables ...")
    classifier = QuizableClassifier()
    try:
        classifier.load()
        print("      -> classifieur entraîné chargé")
    except FileNotFoundError:
        print("      -> aucun classifieur entraîné trouvé, utilisation de l'heuristique")
    results = select_quizable_passages(passages, classifier=classifier, threshold=args.threshold)
    print(f"      -> {len(results)} passage(s) retenu(s)")

    print(f"[3/3] Génération de {args.n_questions} question(s) ...")
    quiz = generate_quiz(results, n_questions=args.n_questions)
    print(f"      -> {len(quiz.questions)} question(s) générée(s), {quiz.skipped} passage(s) ignoré(s)")

    print("\n" + evaluate_quiz_quality(quiz).report())

    output = {
        "questions": [
            {
                "question": q.question,
                "options": q.options,
                "correct_index": q.correct_index,
                "explanation": q.explanation,
                "source_page": q.source_page,
            }
            for q in quiz.questions
        ]
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nQuiz sauvegardé dans {args.output}")
    else:
        print("\n" + json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()