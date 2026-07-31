from __future__ import annotations

import base64
from dataclasses import dataclass

import fitz  # PyMuPDF
import openai
from openai import OpenAI

from .utils import CONFIG, clean_text, split_into_passages

# En dessous de cette taille (en pixels), une image est presque toujours une
# icône ou un logo décoratif -- on ne dépense pas d'appel API dessus.
MIN_IMAGE_DIMENSION = 150
MAX_IMAGES_PER_PAGE = 4  # garde-fou de coût sur des pages très chargées en visuels

_OCR_PROMPT = (
    "Retranscris fidèlement tout le texte visible sur cette image de page de document, "
    "sans rien résumer ni reformuler, dans l'ordre de lecture naturel. "
    "Si aucun texte n'est visible sur l'image, réponds exactement : AUCUN_TEXTE"
)

_IMAGE_DESCRIPTION_PROMPT = (
    "Ce visuel est extrait d'un support de cours. Décris uniquement les éléments "
    "factuels qu'il contient et qui pourraient servir à poser une question de quiz : "
    "légendes, étiquettes, valeurs numériques, étapes d'un processus, relations entre "
    "éléments d'un schéma. Ne décris pas le style visuel ni la mise en forme. "
    "Si ce visuel est purement décoratif (logo, icône, photo sans contenu informatif), "
    "réponds exactement : DECORATIF"
)


@dataclass
class Passage:
    """Un passage de texte extrait du document, avec sa provenance.

    source_type distingue l'origine du contenu : "texte" (extraction directe),
    "image_ocr" (texte lu sur une page scannée) ou "image_description"
    (schéma/graphique décrit par la vision LLM) -- utile pour la démo et
    pour comprendre, en cas de question étrange, si elle vient d'un visuel
    mal interprété plutôt que d'un vrai texte.
    """

    id: str
    text: str
    page: int
    source_type: str = "texte"


def _get_vision_client() -> OpenAI | None:
    """Retourne un client OpenAI si une clé API est configurée, sinon None
    (dégradation silencieuse : le pipeline continue sans vision)."""
    if not CONFIG.openai_api_key:
        return None
    return OpenAI(api_key=CONFIG.openai_api_key)


def _image_to_data_uri(image_bytes: bytes, image_format: str = "png") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{image_format};base64,{b64}"


def _ask_vision(client: OpenAI, data_uri: str, prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=CONFIG.openai_model,
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }],
        )
        return (response.choices[0].message.content or "").strip()
    except openai.OpenAIError as exc:
        print(f"[extraction] appel vision échoué, image ignorée ({exc.__class__.__name__}: {exc})")
        return ""


def _ocr_page(client: OpenAI, page: fitz.Page) -> str:
    """OCR d'une page entière (cas d'une page scannée sans texte sélectionnable)."""
    pix = page.get_pixmap(dpi=150)
    text = _ask_vision(client, _image_to_data_uri(pix.tobytes("png")), _OCR_PROMPT)
    return "" if text == "AUCUN_TEXTE" else text


def _describe_embedded_images(client: OpenAI, doc: fitz.Document, page: fitz.Page) -> list[str]:
    """Décrit les images significatives (schémas, graphiques) intégrées à une page.
    Ignore les petites images (icônes/logos) et limite le nombre d'appels par page."""
    descriptions = []
    images = page.get_images(full=True)[:MAX_IMAGES_PER_PAGE]

    for img in images:
        xref = img[0]
        try:
            base_image = doc.extract_image(xref)
        except Exception:
            continue

        if base_image["width"] < MIN_IMAGE_DIMENSION or base_image["height"] < MIN_IMAGE_DIMENSION:
            continue  # trop petit pour être un schéma informatif

        data_uri = _image_to_data_uri(base_image["image"], base_image["ext"])
        description = _ask_vision(client, data_uri, _IMAGE_DESCRIPTION_PROMPT)

        if description and description != "DECORATIF":
            descriptions.append(description)

    return descriptions


def extract_passages(
    pdf_path: str,
    use_vision: bool = True,
    progress_callback=None,
) -> list[Passage]:
    """Pipeline complet d'extraction : PDF -> texte nettoyé + visuels décrits -> passages.

    use_vision=True (par défaut) active l'OCR des pages scannées et la
    description des schémas/graphiques via l'API OpenAI (coût additionnel :
    un appel par page scannée et par image significative). Passer
    use_vision=False pour une extraction texte pure, sans aucun appel API.

    progress_callback(current_page, total_pages, message) est appelé après
    chaque page traitée -- utile pour afficher une barre de progression côté
    Streamlit, en particulier quand des appels vision ralentissent le traitement.
    """
    doc = fitz.open(pdf_path)
    client = _get_vision_client() if use_vision else None
    total_pages = len(doc)

    passages: list[Passage] = []
    counter = 0

    for page_number, page in enumerate(doc, start=1):
        raw_text = page.get_text("text")
        page_contents: list[tuple[str, str]] = []  # (source_type, texte_brut)
        used_vision = False

        if raw_text.strip():
            page_contents.append(("texte", raw_text))
        elif client is not None:
            used_vision = True
            if progress_callback:
                progress_callback(page_number, total_pages, "page scannée : lecture par OCR (vision)...")
            ocr_text = _ocr_page(client, page)
            if ocr_text:
                page_contents.append(("image_ocr", ocr_text))

        if client is not None:
            n_images = len(page.get_images(full=True))
            if n_images:
                used_vision = True
                if progress_callback:
                    progress_callback(
                        page_number, total_pages,
                        f"analyse de {min(n_images, MAX_IMAGES_PER_PAGE)} image(s) (vision)...",
                    )
            for description in _describe_embedded_images(client, doc, page):
                page_contents.append(("image_description", description))

        for source_type, raw in page_contents:
            cleaned = clean_text(raw)
            if not cleaned:
                continue
            for chunk in split_into_passages(cleaned, CONFIG.min_passage_chars, CONFIG.max_passage_chars):
                counter += 1
                passages.append(Passage(id=f"p{counter:04d}", text=chunk, page=page_number, source_type=source_type))

        if progress_callback and not used_vision:
            progress_callback(page_number, total_pages, "texte extrait directement")

    doc.close()

    if not passages:
        raise ValueError(
            "Aucun contenu exploitable n'a été trouvé dans ce PDF (ni texte, ni image "
            "interprétable)." + ("" if use_vision else " Essaie avec use_vision=True si "
            "ce document est scanné ou contient surtout des schémas.")
        )

    return passages


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.extraction <chemin_vers_pdf>")
        sys.exit(1)

    result = extract_passages(sys.argv[1])
    print(f"{len(result)} passage(s) extrait(s).\n")
    for p in result[:5]:
        print(f"--- {p.id} (page {p.page}, {p.source_type}) ---")
        print(p.text[:300], "...\n")