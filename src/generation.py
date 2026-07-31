from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .classification import ClassificationResult
from .utils import CONFIG

_SYSTEM_PROMPT = """Tu es un générateur de questions de quiz pour un contexte pédagogique.
À partir d'un passage de cours ou de document, tu génères UNE question à choix multiples (QCM).

Règles strictes :
- La question doit porter sur un fait, une définition ou une donnée EXPLICITEMENT présente dans le passage.
- Propose exactement 4 options : 1 bonne réponse + 3 distracteurs.
- Les distracteurs doivent être plausibles (même thème, même registre) mais clairement faux au vu du passage.
- N'invente aucune information absente du passage.
- Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ni après, au format :
{
  "question": "...",
  "options": ["...", "...", "...", "..."],
  "correct_index": 0,
  "explanation": "Courte justification citant l'élément du passage qui justifie la réponse."
}
Si le passage ne contient réellement aucune information exploitable pour une question factuelle,
réponds avec: {"question": null}
"""


@dataclass
class Question:
    question: str
    options: list[str]
    correct_index: int
    explanation: str
    source_passage_id: str
    source_page: int


@dataclass
class Quiz:
    questions: list[Question] = field(default_factory=list)
    skipped: int = 0  # passages pour lesquels aucune question exploitable n'a pu être générée


def _get_client() -> OpenAI:
    CONFIG.check()
    return OpenAI(api_key=CONFIG.openai_api_key)


def _extract_json(raw: str) -> dict:
    """Extrait un objet JSON de la réponse du modèle, en tolérant d'éventuels
    fences ```json ... ``` malgré la consigne de sortie stricte."""
    raw = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else raw
    return json.loads(candidate)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_model(client: OpenAI, passage_text: str) -> dict:
    response = client.chat.completions.create(
        model=CONFIG.openai_model,
        max_tokens=500,
        response_format={"type": "json_object"},  # force une sortie JSON valide côté API
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Passage :\n\"\"\"\n{passage_text}\n\"\"\""},
        ],
    )
    text_block = response.choices[0].message.content or ""
    return _extract_json(text_block)


def generate_question_from_passage(
    client: OpenAI,
    result: ClassificationResult,
) -> Question | None:
    """Génère une question à partir d'un passage classifié comme quizable.
    Retourne None si le modèle juge le passage finalement inexploitable
    (garde-fou en plus de la classification amont)."""
    payload = _call_model(client, result.passage.text)

    if not payload.get("question"):
        return None

    options = payload["options"]
    if len(options) != 4 or not (0 <= payload["correct_index"] < 4):
        # Réponse mal formée malgré la consigne : on l'écarte plutôt que de planter la démo
        return None

    return Question(
        question=payload["question"],
        options=options,
        correct_index=payload["correct_index"],
        explanation=payload.get("explanation", ""),
        source_passage_id=result.passage.id,
        source_page=result.passage.page,
    )


def generate_quiz(
    results: list[ClassificationResult],
    n_questions: int,
    progress_callback=None,
) -> Quiz:
    """Génère un quiz complet à partir des passages quizables (déjà triés
    par score de pertinence par `select_quizable_passages`).

    `progress_callback(current, total)` est appelé après chaque tentative,
    utile pour une barre de progression côté Streamlit.
    """
    client = _get_client()
    quiz = Quiz()
    candidates = results[: max(n_questions * 2, n_questions + 3)]  # marge pour les échecs de génération

    for i, result in enumerate(candidates, start=1):
        if len(quiz.questions) >= n_questions:
            break
        try:
            question = generate_question_from_passage(client, result)
        except (json.JSONDecodeError, openai.OpenAIError) as exc:
            question = None
            quiz.skipped += 1
            print(f"[generation] passage {result.passage.id} ignoré ({exc.__class__.__name__}: {exc})")

        if question is not None:
            quiz.questions.append(question)
        else:
            quiz.skipped += 1

        if progress_callback:
            progress_callback(i, len(candidates))

    return quiz