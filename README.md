# 📚 RéviseIA — Génération automatique de quiz à partir d'un document PDF

Application web permettant de transformer un document de cours (PDF) en questionnaire à choix multiples (QCM) exploitable pour la révision, en combinant traitement automatique du langage naturel (NLP), apprentissage automatique classique et un grand modèle de langage (LLM).

🔗 **Démo en ligne :** [https://quiz-rna.onrender.com/](https://quiz-rna.onrender.com/)
📦 **Dépôt Git :** [https://github.com/setraniainabruno/pdf_to_quiz.git](https://github.com/setraniainabruno/pdf_to_quiz.git)

> Projet réalisé dans le cadre du Master 2 — Mention Informatique, Parcours Sciences de Données et Intelligence Artificielle (SDIA), **Université de Fianarantsoa / EMIT** — Projet RNA, NLP (Année universitaire 2025–2026).

## 📋 Contexte

L'objectif est de concevoir une application capable de transformer un document de cours (PDF) en QCM exploitable pour la révision, en combinant des techniques de NLP et d'apprentissage automatique classique, ainsi que l'usage d'un LLM via une API externe pour la génération de texte.

## ✨ Fonctionnalités

- Génération d'un quiz à partir de n'importe quel PDF de cours (en français)
- Questionnaire à choix multiples (QCM) : 4 options par question, une seule correcte, avec explication
- Gestion des documents scannés (OCR) et des schémas/images (description automatique via vision)
- Sélection intelligente des passages pertinents (classification "quizable" / "non quizable")
- Correction interactive et score final dans l'interface
- Annotation manuelle de passages pour enrichir le jeu de données
- Évaluation humaine de la qualité des questions générées (pertinence, clarté, plausibilité des distracteurs)

## 🛠️ Choix méthodologiques

| Choix | Décision retenue |
|---|---|
| Type de questions | QCM (4 options, 1 correcte, 3 distracteurs) |
| Langue | Français |
| Génération | Prompting d'un LLM (zero/few-shot), pas de fine-tuning |
| Approche pédagogique | Pipeline NLP classique : prétraitement → représentation → modélisation → post-traitement |

Le choix du prompting plutôt que du fine-tuning s'explique par l'absence d'infrastructure GPU dédiée, la nécessité de traiter du français sans corpus francophone dédié, et la simplicité d'un pipeline en un seul appel structuré (question + distracteurs cohérents).

## 🏗️ Architecture du pipeline

```
PDF (cours) → Extraction → Classification → Génération LLM → Quiz QCM
                                                    ↓
                                    Évaluation (métriques + éval. humaine)
```

### Modules Python

| Module | Rôle |
|---|---|
| `src/extraction.py` | Bloc 1 : extraction du texte du PDF → passages |
| `src/classification.py` | Bloc 2 : sélection des passages "quizables" |
| `src/generation.py` | Bloc 3 : génération des questions via LLM |
| `src/evaluation.py` | Bloc 4 : métriques de classification et contrôles qualité |
| `src/human_eval.py` | Évaluation humaine des questions générées |
| `src/dataset.py` | Gestion du jeu de données annoté |
| `src/utils.py` | Configuration, nettoyage de texte |
| `app/streamlit_app.py` | Interface web (3 modes) |
| `scripts/train_classifier.py` | Entraînement et évaluation du classifieur |

### 1. Extraction (`extraction.py`)

Extraction du texte page par page avec **PyMuPDF (fitz)**, nettoyage (recollage de mots coupés, normalisation), puis segmentation en passages de 200 à 1200 caractères. Gestion des pages scannées (OCR via la vision de l'API OpenAI) et des images/schémas significatifs (description automatique), avec traçabilité de l'origine de chaque passage (`source_type` : `texte`, `image_ocr`, `image_description`).

### 2. Classification (`classification.py`)

Détermine quels passages contiennent assez d'information factuelle pour générer une bonne question, via :
- un score heuristique (dates, chiffres, marqueurs de définition, noms propres) ;
- un classifieur binaire supervisé (embeddings OpenAI `text-embedding-3-small` + régression logistique **implémentée from scratch en numpy**, sans scikit-learn/scipy, pour des raisons de compatibilité environnementale).

### 3. Génération (`generation.py`)

Appel à l'API OpenAI (modèle `gpt-4.1-mini` par défaut) avec un prompt structuré et sortie JSON stricte (question, 4 options, index de la bonne réponse, explication), en un seul appel garantissant la cohérence question/distracteurs. Gestion des échecs transitoires via `tenacity`.

### 4. Évaluation (`evaluation.py`, `human_eval.py`)

Trois niveaux d'évaluation :
- **Métriques de classification** (accuracy, precision, recall, F1, matrice de confusion) sur le classifieur de passages ;
- **Contrôles structurels** du quiz généré (options distinctes, longueur, couverture du document) ;
- **Évaluation humaine** (pertinence, clarté, plausibilité des distracteurs, notées sur 5) ;
- BLEU-1 / ROUGE-L implémentés (optionnels, nécessitent un jeu de référence non encore constitué).

## 🖥️ Interface — RéviseIA (Streamlit)

Trois modes accessibles depuis la barre latérale :
1. **Générer un quiz** — mode principal pour l'utilisateur final
2. **Annoter des passages** — construction du jeu de données
3. **Évaluer la qualité** — évaluation humaine des questions générées

Identité visuelle dédiée (typographies Fraunces & Manrope, palette encre marine / accent ambre).

## 🧰 Pile technique

| Bibliothèque | Rôle |
|---|---|
| PyMuPDF (fitz) | Extraction du texte des PDF |
| openai | Génération, embeddings, vision (OCR / description) |
| numpy | Régression logistique, métriques |
| streamlit | Interface web |
| joblib | Sérialisation du classifieur |
| tenacity | Nouvelles tentatives sur échec d'appel API |
| pytest | Tests unitaires (19 tests) |

## 📊 Jeu de données

- **1276** passages annotés, issus de **4** documents sources distincts
- **606** passages "quizables" / **670** "non quizables"
- Longueur moyenne d'un passage : **839** caractères
- **3%** annoté manuellement, **97%** annoté automatiquement par le LLM (traçable via `annotated_by`)
- Découpage stratifié 80% entraînement / 20% validation
- Format : JSONL (`data/annotations.jsonl`)

## 📈 Résultats

**Métriques de classification (jeu de validation) :**

| Métrique | Valeur |
|---|---|
| Accuracy | 0.694 |
| Precision | 0.684 |
| Recall | 0.661 |
| F1-score | 0.672 |

**Évaluation humaine (notes sur 5) :**

| Critère | Note moyenne |
|---|---|
| Pertinence | 3.40 |
| Clarté | 4.80 |
| Plausibilité des distracteurs | 3.20 |

## 🚀 Installation et lancement

```bash
git clone https://github.com/setraniainabruno/pdf_to_quiz.git
cd projetRNA
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

L'application est accessible sur `http://localhost:8501`.

Une clé API OpenAI valide doit être configurée (variable d'environnement) pour activer la génération, la classification par embeddings et l'analyse des images.

## 📂 Structure du dépôt

```
projetRNA/
├── app/streamlit_app.py       # Interface web (3 modes)
├── src/
│   ├── extraction.py          # Bloc 1 : PDF -> passages
│   ├── classification.py      # Bloc 2 : sélection des passages
│   ├── generation.py          # Bloc 3 : génération des questions
│   ├── evaluation.py          # Bloc 4 : métriques
│   ├── human_eval.py          # Évaluation humaine
│   ├── dataset.py             # Gestion du jeu de données
│   └── utils.py                # Configuration, nettoyage de texte
├── scripts/train_classifier.py # Entraînement + évaluation
├── tests/test_pipeline.py      # 19 tests unitaires (pytest)
├── data/                       # Jeu de données annoté (non versionné)
├── requirements.txt
└── README.md
```
