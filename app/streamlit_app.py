from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Permet d'importer le package `src` depuis app/, quel que soit le
# répertoire courant depuis lequel `streamlit run` est lancé.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification import QuizableClassifier, select_quizable_passages  # noqa: E402
from src.dataset import (  # noqa: E402
    Annotation,
    already_annotated_ids,
    append_annotation,
    corpus_statistics,
    load_annotations,
)
from src.evaluation import evaluate_quiz_quality  # noqa: E402
from src.extraction import extract_passages  # noqa: E402
from src.generation import Quiz, generate_quiz  # noqa: E402
from src.human_eval import (  # noqa: E402
    CRITERIA_LABELS,
    HumanRating,
    aggregate_ratings,
    append_rating,
    load_ratings,
)
from src.utils import CONFIG  # noqa: E402

st.set_page_config(page_title="RéviseIA — Quiz à partir de vos cours", page_icon="🎓", layout="centered")


# --------------------------------------------------------------------------- #
# Identité visuelle : polices, couleurs, cartes, boutons (cf. design brief)
# --------------------------------------------------------------------------- #

def inject_theme() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --ink: #1B2A4A;
        --ink-soft: #4A5A7C;
        --paper: #F7F8FB;
        --surface: #FFFFFF;
        --amber: #E8A33D;
        --amber-dark: #C97F1E;
        --blue: #3E7CB1;
        --green: #2F9E44;
        --border: #E2E5EC;
    }

    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }

    .stApp { background: var(--paper); }

    h1, h2, h3 {
        font-family: 'Fraunces', Georgia, serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.01em;
        font-weight: 600 !important;
    }

    p, li, label, .stMarkdown { color: var(--ink-soft); }

    /* ---- Bandeau héros ---- */
    .hero {
        background: linear-gradient(135deg, var(--ink) 0%, #26386A 100%);
        border-radius: 20px;
        padding: 2.4rem 2.2rem 2.1rem 2.2rem;
        margin-bottom: 1.7rem;
        box-shadow: 0 8px 24px rgba(27,42,74,0.18);
    }
    .hero-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase;
        letter-spacing: .14em;
        font-size: .72rem;
        color: var(--amber);
        margin-bottom: .7rem;
        font-weight: 500;
    }
    .hero h1 {
        font-size: 2.5rem !important;
        color: #FFFFFF !important;
        margin: 0 0 .6rem 0 !important;
        line-height: 1.08;
    }
    .hero .highlight {
        background-image: linear-gradient(120deg, rgba(232,163,61,.55) 0%, rgba(232,163,61,.55) 100%);
        background-repeat: no-repeat;
        background-size: 100% 38%;
        background-position: 0 88%;
        padding: 0 .1rem;
    }
    .hero p {
        font-size: 1.04rem;
        color: #C9D2E3;
        max-width: 42rem;
        margin: 0;
        line-height: 1.55;
    }

    /* ---- Boutons ---- */
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px !important;
        font-weight: 700 !important;
        border: 1px solid var(--border) !important;
        padding: 0.55rem 1.1rem !important;
        transition: transform .05s ease-in;
    }
    .stButton > button:active { transform: scale(0.98); }
    .stButton > button[kind="primary"],
    button[data-testid="baseButton-primary"],
    button[data-testid="stBaseButton-primary"] {
        background: var(--amber) !important;
        color: var(--ink) !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background: var(--amber-dark) !important;
        color: #FFFFFF !important;
    }

    /* ---- Cartes (conteneurs à bordure) ---- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        border: 1px solid var(--border) !important;
        background: var(--surface) !important;
        box-shadow: 0 1px 3px rgba(27,42,74,0.05);
    }

    /* ---- Barre latérale ---- */
    section[data-testid="stSidebar"] { background: var(--ink); }
    section[data-testid="stSidebar"] * { color: #E9EDF6 !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #FFFFFF !important; font-family: 'Fraunces', serif !important; }
    section[data-testid="stSidebar"] .stRadio > label { font-weight: 600; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }

    .sidebar-brand {
        font-family: 'Fraunces', serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: .1rem;
    }
    .sidebar-brand span { color: var(--amber); }
    .sidebar-tagline {
        font-family: 'IBM Plex Mono', monospace;
        font-size: .72rem;
        color: #9FB0D1;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin-bottom: 1.1rem;
    }

    /* ---- Barre de progression ---- */
    .stProgress > div > div > div > div { background-color: var(--amber) !important; }

    /* ---- Métriques (score) ---- */
    div[data-testid="stMetricValue"] { font-family: 'Fraunces', serif; color: var(--ink); font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-family: 'IBM Plex Mono', monospace; text-transform: uppercase; font-size: .7rem !important; letter-spacing: .06em; }

    /* ---- Uploader ---- */
    section[data-testid="stFileUploaderDropzone"] {
        border-radius: 12px !important;
        border: 1.5px dashed var(--border) !important;
        background: var(--surface) !important;
    }
    </style>
    """, unsafe_allow_html=True)


inject_theme()


# --------------------------------------------------------------------------- #
# État de session (préfixé par mode pour éviter toute interférence)
# --------------------------------------------------------------------------- #

def init_state() -> None:
    defaults = {
        # Mode "Générer un quiz"
        "quiz_result": None,        # objet Quiz généré
        "quiz_answers": {},         # {index_question: index_option_choisi}
        "quiz_validated": False,
        "quiz_quality_report": None,
        "quiz_source_doc": None,
        # Mode "Annoter"
        "annotate_passages": [],           # passages restants à annoter pour le doc en cours
        "annotate_doc_name": None,
        "annotate_n_labeled_session": 0,
        # Mode "Évaluer la qualité"
        "eval_rated_indices": set(),       # index des questions déjà notées dans le quiz courant
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_quiz() -> None:
    st.session_state.quiz_result = None
    st.session_state.quiz_answers = {}
    st.session_state.quiz_validated = False
    st.session_state.quiz_quality_report = None
    st.session_state.quiz_source_doc = None
    st.session_state.eval_rated_indices = set()


init_state()


# --------------------------------------------------------------------------- #
# En-tête + vérification de la clé API (commune aux deux modes)
# --------------------------------------------------------------------------- #

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Cours PDF → quiz en 2 minutes</div>
    <h1>Transformez vos cours en <span class="highlight">quiz interactifs</span></h1>
    <p>Déposez un PDF, l'IA repère ce qui compte vraiment et génère un QCM prêt à réviser —
    schémas et pages scannées compris.</p>
</div>
""", unsafe_allow_html=True)

if not CONFIG.openai_api_key:
    st.error(
        "⚠️ Clé API OpenAI manquante. Crée un fichier `.env` à la racine du projet "
        "contenant :\n\n`OPENAI_API_KEY=sk-...`\n\npuis relance l'app.",
        icon="🚫",
    )
    st.stop()


# --------------------------------------------------------------------------- #
# Barre latérale : sélecteur de mode + paramètres propres à chaque mode
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">Révise<span>IA</span></div>
    <div class="sidebar-tagline">Vos cours, en mode révision</div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Mode",
        ["🧠 Générer un quiz", "🏷️ Annoter des passages", "🧪 Évaluer la qualité"],
        help="« Générer un quiz » est la démo finale. « Annoter » sert à construire "
             "le jeu de données pour entraîner le classifieur (bloc 2). "
             "« Évaluer la qualité » sert à noter humainement les questions générées (bloc 3/4).",
    )
    st.divider()

    if mode == "🧠 Générer un quiz":
        st.header("Paramètres")
        n_questions = st.slider("Nombre de questions", min_value=3, max_value=15, value=5)
        analyze_images = st.checkbox(
            "🖼️ Analyser aussi les images (OCR + description des schémas)",
            value=True,
            help="Utilise la vision de l'API OpenAI pour lire le texte des pages scannées et "
                 "décrire les schémas/graphiques. Ajoute un appel API par page scannée et par "
                 "image significative -- décoche pour une extraction texte pure, sans coût additionnel.",
        )
        threshold = st.slider(
            "Seuil de sélection des passages",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="Plus le seuil est élevé, plus les passages retenus sont sélectifs "
                 "(moins nombreux mais plus riches).",
        )
        st.divider()
        st.caption(
            "Classification : "
            + ("modèle entraîné chargé ✅" if Path(CONFIG.classifier_path).exists()
               else "heuristique (pas de modèle entraîné) ℹ️")
        )
    elif mode == "🏷️ Annoter des passages":
        st.header("📊 Corpus annoté")
        stats = corpus_statistics(load_annotations())
        st.metric("Passages annotés", stats["n_passages"])
        col1, col2 = st.columns(2)
        col1.metric("Quizables", stats["n_quizable"])
        col2.metric("Non quizables", stats["n_non_quizable"])
        st.caption(f"Documents distincts : {stats['n_documents']}")
        st.caption(f"Longueur moyenne : {stats['avg_passage_length']:.0f} caractères")

        if stats["n_passages"] < 40:
            st.warning(f"Objectif recommandé : ≥ 80 passages (actuellement {stats['n_passages']}).")
        else:
            st.success(f"{stats['n_passages']} passages : de quoi entraîner un premier classifieur.")

        st.divider()
        annotations_path = PROJECT_ROOT / "data" / "annotations.jsonl"
        if annotations_path.exists():
            st.download_button(
                "⬇️ Télécharger annotations.jsonl",
                data=annotations_path.read_bytes(),
                file_name="annotations.jsonl",
                mime="application/jsonl",
            )

    else:  # 🧪 Évaluer la qualité
        st.header("📊 Évaluations humaines")
        eval_report = aggregate_ratings(load_ratings())
        st.metric("Questions évaluées", eval_report.n_ratings)
        if eval_report.n_ratings > 0:
            st.caption(f"Pertinence moyenne : {eval_report.avg_pertinence:.1f} / 5")
            st.caption(f"Clarté moyenne : {eval_report.avg_clarte:.1f} / 5")
            st.caption(f"Distracteurs plausibles (moy) : {eval_report.avg_distracteurs_plausibles:.1f} / 5")
            if eval_report.pct_faible_qualite > 0:
                st.warning(f"{eval_report.pct_faible_qualite:.0%} des questions ont au moins un critère faible (≤2).")

        st.divider()
        eval_path = PROJECT_ROOT / "data" / "human_eval.jsonl"
        if eval_path.exists():
            st.download_button(
                "⬇️ Télécharger human_eval.jsonl",
                data=eval_path.read_bytes(),
                file_name="human_eval.jsonl",
                mime="application/jsonl",
            )


# =============================================================================
# MODE 1 : Générer un quiz (démo finale)
# =============================================================================

def render_quiz_mode() -> None:
    st.caption("Dépose un PDF de cours ou d'article : l'app extrait le contenu, "
               "sélectionne les passages pertinents et génère un QCM.")

    uploaded_file = st.file_uploader("Document PDF", type=["pdf"], on_change=reset_quiz, key="quiz_uploader")

    if uploaded_file is not None and st.session_state.quiz_result is None:
        if st.button("🚀 Générer le quiz", type="primary"):
            tmp_path = PROJECT_ROOT / "app" / "_tmp_upload.pdf"
            tmp_path.write_bytes(uploaded_file.getvalue())

            try:
                extraction_progress = st.progress(0.0, text="Extraction du texte du PDF...")

                def _on_extraction_progress(current: int, total: int, message: str) -> None:
                    extraction_progress.progress(
                        current / total, text=f"Page {current}/{total} — {message}"
                    )

                passages = extract_passages(
                    str(tmp_path), use_vision=analyze_images, progress_callback=_on_extraction_progress
                )
                extraction_progress.empty()
                st.success(f"{len(passages)} passage(s) extrait(s) du document.")

                with st.spinner("Sélection des passages pertinents..."):
                    classifier = QuizableClassifier()
                    if Path(CONFIG.classifier_path).exists():
                        classifier.load()
                    results = select_quizable_passages(
                        passages, classifier=classifier, threshold=threshold, top_k=n_questions * 3
                    )
                if not results:
                    st.warning(
                        "Aucun passage jugé assez informatif pour générer une question. "
                        "Essaie de baisser le seuil dans la barre latérale."
                    )
                    st.stop()
                st.success(f"{len(results)} passage(s) retenu(s) comme pertinents.")

                progress_bar = st.progress(0.0, text="Génération des questions...")

                def _on_progress(current: int, total: int) -> None:
                    progress_bar.progress(current / total, text=f"Génération des questions... ({current}/{total})")

                quiz = generate_quiz(results, n_questions=n_questions, progress_callback=_on_progress)
                progress_bar.empty()

                if not quiz.questions:
                    st.error("Le modèle n'a pas réussi à générer de question exploitable sur ce document.")
                    st.stop()

                st.session_state.quiz_result = quiz
                st.session_state.quiz_quality_report = evaluate_quiz_quality(quiz)
                st.session_state.quiz_source_doc = uploaded_file.name

            except ValueError as exc:
                st.error(str(exc))
            finally:
                tmp_path.unlink(missing_ok=True)

            st.rerun()

    if st.session_state.quiz_result is not None:
        _render_quiz(st.session_state.quiz_result)


def _render_quiz(quiz: Quiz) -> None:
    st.divider()
    st.subheader(f"Quiz généré — {len(quiz.questions)} question(s)")

    for i, q in enumerate(quiz.questions):
        with st.container(border=True):
            st.markdown(f"**{i + 1}. {q.question}**")
            choice = st.radio(
                label=f"question_{i}",
                options=list(range(len(q.options))),
                format_func=lambda idx, opts=q.options: opts[idx],
                key=f"radio_{i}",
                label_visibility="collapsed",
                disabled=st.session_state.quiz_validated,
            )
            st.session_state.quiz_answers[i] = choice

            if st.session_state.quiz_validated:
                if choice == q.correct_index:
                    st.success(f"✅ Correct — {q.explanation}")
                else:
                    st.error(f"❌ Réponse attendue : « {q.options[q.correct_index]} » — {q.explanation}")
                st.caption(f"Source : page {q.source_page} du document")
        st.write("")

    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.quiz_validated:
            if st.button("✔️ Valider mes réponses", type="primary"):
                st.session_state.quiz_validated = True
                st.rerun()
        else:
            score = sum(
                1 for i, q in enumerate(quiz.questions) if st.session_state.quiz_answers.get(i) == q.correct_index
            )
            st.metric("Score", f"{score} / {len(quiz.questions)}")

    with col2:
        if st.button("🔄 Nouveau quiz"):
            reset_quiz()
            st.rerun()

    with st.expander("📊 Rapport qualité (analyse des résultats)"):
        st.text(st.session_state.quiz_quality_report.report())


# =============================================================================
# MODE 2 : Annoter des passages (construction du jeu de données)
# =============================================================================

def render_annotate_mode() -> None:
    st.caption(
        "Pour chaque passage, indique s'il contient assez d'information factuelle "
        "(fait, définition, date, chiffre) pour qu'une bonne question de quiz en soit extraite. "
        "Objectif recommandé : 80-150 passages, en variant les documents sources."
    )

    uploaded_file = st.file_uploader("Document PDF à annoter", type=["pdf"], key="annotate_uploader")
    analyze_images = st.checkbox(
        "🖼️ Analyser aussi les images (OCR + description des schémas)",
        value=True,
        key="annotate_analyze_images",
        help="Utilise la vision de l'API OpenAI pour lire le texte des pages scannées et "
             "décrire les schémas/graphiques.",
    )

    if uploaded_file is not None and uploaded_file.name != st.session_state.annotate_doc_name:
        tmp_path = PROJECT_ROOT / "app" / "_tmp_annotate.pdf"
        tmp_path.write_bytes(uploaded_file.getvalue())

        try:
            extraction_progress = st.progress(0.0, text="Extraction du texte du PDF...")

            def _on_extraction_progress(current: int, total: int, message: str) -> None:
                extraction_progress.progress(current / total, text=f"Page {current}/{total} — {message}")

            all_passages = extract_passages(
                str(tmp_path), use_vision=analyze_images, progress_callback=_on_extraction_progress
            )
            extraction_progress.empty()
        except ValueError as exc:
            st.error(str(exc))
            all_passages = []
        finally:
            tmp_path.unlink(missing_ok=True)

        already_done = already_annotated_ids()
        remaining = [p for p in all_passages if (uploaded_file.name, p.id) not in already_done]

        st.session_state.annotate_passages = remaining
        st.session_state.annotate_doc_name = uploaded_file.name
        st.session_state.annotate_n_labeled_session = 0

        skipped = len(all_passages) - len(remaining)
        if skipped:
            st.info(f"{skipped} passage(s) de ce document déjà annoté(s) précédemment, ignoré(s).")

    def _label_current(label: int) -> None:
        passage = st.session_state.annotate_passages.pop(0)
        append_annotation(
            Annotation(
                passage_id=passage.id,
                text=passage.text,
                source_doc=st.session_state.annotate_doc_name,
                page=passage.page,
                label=label,
                annotated_by="manuel",
                source_type=passage.source_type,
            )
        )
        st.session_state.annotate_n_labeled_session += 1

    if st.session_state.annotate_passages:
        passage = st.session_state.annotate_passages[0]
        remaining_count = len(st.session_state.annotate_passages)

        st.progress(
            st.session_state.annotate_n_labeled_session
            / max(1, st.session_state.annotate_n_labeled_session + remaining_count),
            text=f"{remaining_count} passage(s) restant(s) dans ce document",
        )

        source_badge = {"texte": "📄 texte", "image_ocr": "🖼️ OCR", "image_description": "🖼️ schéma décrit"}.get(
            passage.source_type, passage.source_type
        )
        st.markdown(f"**Page {passage.page}** — {source_badge}")
        st.info(passage.text)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Quizable", type="primary", use_container_width=True):
                _label_current(1)
                st.rerun()
        with col2:
            if st.button("❌ Non quizable", use_container_width=True):
                _label_current(0)
                st.rerun()
        with col3:
            if st.button("⏭️ Passer (ne pas annoter)", use_container_width=True):
                st.session_state.annotate_passages.pop(0)
                st.rerun()

    elif st.session_state.annotate_doc_name is not None:
        st.success(
            f"Document « {st.session_state.annotate_doc_name} » entièrement annoté "
            f"({st.session_state.annotate_n_labeled_session} passage(s) labellisé(s) dans cette session). "
            "Dépose un autre PDF pour continuer."
        )
    else:
        st.info("Dépose un PDF ci-dessus pour commencer à annoter.")


# =============================================================================
# MODE 3 : Évaluer la qualité (notation humaine des questions générées)
# =============================================================================

def render_eval_mode() -> None:
    st.caption(
        "Note chaque question du dernier quiz généré sur 3 critères indépendants. "
        "C'est la mesure de qualité recommandée tant qu'aucun jeu de questions de "
        "référence n'existe (voir evaluation.py pour le détail)."
    )

    quiz = st.session_state.quiz_result
    if quiz is None or not quiz.questions:
        st.info(
            "Aucun quiz à évaluer pour l'instant. Va d'abord dans « 🧠 Générer un quiz », "
            "génère un quiz, puis reviens ici pour le noter."
        )
        return

    st.caption(f"Document source : « {st.session_state.quiz_source_doc} »")

    for i, q in enumerate(quiz.questions):
        already_rated = i in st.session_state.eval_rated_indices
        with st.container(border=True):
            st.markdown(f"**{i + 1}. {q.question}**")
            st.caption(" / ".join(q.options) + f"  → correcte : « {q.options[q.correct_index]} »")

            if already_rated:
                st.success("✅ Déjà évaluée.")
                continue

            scores = {}
            for criterion, label in CRITERIA_LABELS.items():
                scores[criterion] = st.slider(label, min_value=1, max_value=5, value=3, key=f"eval_{criterion}_{i}")
            comment = st.text_input("Commentaire (optionnel)", key=f"eval_comment_{i}")

            if st.button("💾 Enregistrer cette évaluation", key=f"eval_save_{i}"):
                append_rating(
                    HumanRating(
                        question_text=q.question,
                        source_passage_id=q.source_passage_id,
                        source_doc=st.session_state.quiz_source_doc or "inconnu",
                        pertinence=scores["pertinence"],
                        clarte=scores["clarte"],
                        distracteurs_plausibles=scores["distracteurs_plausibles"],
                        commentaire=comment,
                    )
                )
                st.session_state.eval_rated_indices.add(i)
                st.rerun()

    if len(st.session_state.eval_rated_indices) == len(quiz.questions):
        st.divider()
        st.success("Toutes les questions de ce quiz ont été évaluées 🎉")
        with st.expander("📊 Rapport agrégé (toutes évaluations confondues)"):
            st.text(aggregate_ratings(load_ratings()).report())


# --------------------------------------------------------------------------- #
# Aiguillage vers le mode sélectionné
# --------------------------------------------------------------------------- #

if mode == "🧠 Générer un quiz":
    render_quiz_mode()
elif mode == "🏷️ Annoter des passages":
    render_annotate_mode()
else:
    render_eval_mode()