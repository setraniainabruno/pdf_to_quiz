from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DEFAULT_DATASET_PATH = "data/annotations.jsonl"


@dataclass
class Annotation:
    passage_id: str
    text: str
    source_doc: str
    page: int
    label: int  # 1 = quizable, 0 = non quizable
    annotated_by: str = "manuel"
    source_type: str = "texte"  # "texte", "image_ocr" ou "image_description" (cf. extraction.py)


def append_annotation(annotation: Annotation, path: str = DEFAULT_DATASET_PATH) -> None:
    """Ajoute une annotation au fichier JSONL (créé si besoin)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(annotation), ensure_ascii=False) + "\n")


def load_annotations(path: str = DEFAULT_DATASET_PATH) -> list[Annotation]:
    """Charge toutes les annotations. Retourne une liste vide si le fichier n'existe pas encore."""
    file_path = Path(path)
    if not file_path.exists():
        return []

    annotations = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                annotations.append(Annotation(**json.loads(line)))
    return annotations


def already_annotated_ids(path: str = DEFAULT_DATASET_PATH) -> set[tuple[str, str]]:
    """Couples (source_doc, passage_id) déjà annotés.

    Important : passage_id seul ne suffit pas, car il est régénéré à partir
    de "p0001", "p0002", ... à CHAQUE extraction (voir extraction.py) --
    deux documents différents peuvent donc produire des passages portant le
    même id. On combine avec source_doc pour éviter de sauter par erreur un
    passage d'un nouveau document sous prétexte qu'un AUTRE document avait
    déjà un passage "p0001".
    """
    return {(a.source_doc, a.passage_id) for a in load_annotations(path)}


def to_training_arrays(annotations: list[Annotation]) -> tuple[list[str], np.ndarray]:
    texts = [a.text for a in annotations]
    labels = np.array([a.label for a in annotations])
    return texts, labels


def corpus_statistics(annotations: list[Annotation]) -> dict:
    """Statistiques descriptives attendues par le cadrage du projet
    (section "Jeu de données" : taille, prétraitement, répartition)."""
    if not annotations:
        return {
            "n_passages": 0,
            "n_documents": 0,
            "n_quizable": 0,
            "n_non_quizable": 0,
            "avg_passage_length": 0.0,
            "pct_manuel": 0.0,
        }

    n_quizable = sum(1 for a in annotations if a.label == 1)
    n_manuel = sum(1 for a in annotations if a.annotated_by == "manuel")

    return {
        "n_passages": len(annotations),
        "n_documents": len({a.source_doc for a in annotations}),
        "n_quizable": n_quizable,
        "n_non_quizable": len(annotations) - n_quizable,
        "avg_passage_length": float(np.mean([len(a.text) for a in annotations])),
        "pct_manuel": n_manuel / len(annotations),
    }


def train_val_split(
    annotations: list[Annotation], val_ratio: float = 0.2, seed: int = 42
) -> tuple[list[Annotation], list[Annotation]]:
    """Découpage train/validation stratifié (préserve la proportion de
    chaque classe dans les deux sous-ensembles), sans dépendance à
    scikit-learn (cf. classification.py pour le contexte de cette contrainte).
    """
    rng = np.random.default_rng(seed)

    positives = [a for a in annotations if a.label == 1]
    negatives = [a for a in annotations if a.label == 0]

    def _split(group: list[Annotation]) -> tuple[list[Annotation], list[Annotation]]:
        indices = rng.permutation(len(group))
        n_val = max(1, int(len(group) * val_ratio)) if group else 0
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        return [group[i] for i in train_idx], [group[i] for i in val_idx]

    train_pos, val_pos = _split(positives)
    train_neg, val_neg = _split(negatives)

    train = train_pos + train_neg
    val = val_pos + val_neg
    rng.shuffle(train)  # np.random.shuffle voudrait un ndarray ; on mélange les index à la place
    train = [train[i] for i in rng.permutation(len(train))] if train else train
    val = [val[i] for i in rng.permutation(len(val))] if val else val

    return train, val