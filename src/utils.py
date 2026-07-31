from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    """Paramètres globaux du pipeline, centralisés ici pour éviter les
    valeurs en dur éparpillées dans les autres modules."""

    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))

    # Segmentation
    min_passage_chars: int = 200          # en dessous, un passage est jugé trop pauvre pour générer une question
    max_passage_chars: int = 1200         # au-dessus, on découpe (sinon le LLM générerait une question trop vague)

    # Classification "quizable / non quizable"
    classifier_path: str = "src/models/quizable_classifier.joblib"

    def check(self) -> None:
        if not self.openai_api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY manquante. Ajoute-la dans un fichier .env à la racine du projet :\n"
                "OPENAI_API_KEY=sk-..."
            )


CONFIG = Config()


# --------------------------------------------------------------------------- #
# Nettoyage de texte
# --------------------------------------------------------------------------- #

def clean_text(text: str) -> str:
    """Nettoie un texte extrait de PDF : espaces multiples, sauts de ligne
    parasites, tirets de coupure de mots en fin de ligne, etc."""
    if not text:
        return ""

    # Recolle les mots coupés par un tiret en fin de ligne : "informa-\ntion" -> "information"
    text = re.sub(r"-\s*\n\s*", "", text)

    # Remplace les sauts de ligne simples (retour à la ligne dans un même
    # paragraphe) par un espace, mais garde les doubles sauts de ligne
    # (séparateurs de paragraphe)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Compacte les espaces multiples
    text = re.sub(r"[ \t]+", " ", text)

    # Compacte les sauts de ligne multiples en un seul séparateur de paragraphe
    text = re.sub(r"\n{2,}", "\n\n", text)

    return text.strip()


# --------------------------------------------------------------------------- #
# Segmentation en passages
# --------------------------------------------------------------------------- #

def split_into_passages(
    text: str,
    min_chars: int = CONFIG.min_passage_chars,
    max_chars: int = CONFIG.max_passage_chars,
) -> list[str]:
    """Découpe un texte nettoyé en passages cohérents (approximativement
    des paragraphes), en fusionnant les paragraphes trop courts et en
    re-découpant ceux qui sont trop longs.

    On reste volontairement simple (règles + longueur) plutôt que d'utiliser
    un modèle de segmentation sémantique : c'est suffisant pour ce projet et
    ça garde le pipeline explicable, ce qui est valorisé dans l'analyse des
    résultats.
    """
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    passages: list[str] = []
    buffer = ""

    for para in raw_paragraphs:
        candidate = f"{buffer} {para}".strip() if buffer else para

        if len(candidate) < min_chars:
            # Trop court pour être un passage autonome : on accumule
            buffer = candidate
            continue

        if len(candidate) <= max_chars:
            passages.append(candidate)
            buffer = ""
        else:
            # Le paragraphe (ou l'accumulation) dépasse la taille max :
            # on découpe en sous-blocs par phrases
            passages.extend(_split_long_text(candidate, max_chars))
            buffer = ""

    if buffer and len(buffer) >= min_chars // 2:
        # On garde le reliquat s'il n'est pas dérisoire, plutôt que de le perdre
        passages.append(buffer)

    return passages


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Découpe un texte trop long en blocs <= max_chars, sans couper une phrase en deux."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks