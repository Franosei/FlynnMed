# backend/utils.py

from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF


def extract_text_from_pdf(file_path: Path) -> str:
    """
    Extracts and returns full text content from a PDF file using PyMuPDF.

    Args:
        file_path (Path): Path to the PDF file.

    Returns:
        str: Extracted plain text content.
    """
    if not file_path.exists() or file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Invalid PDF file: {file_path}")

    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def extract_text_from_pdf_bytes(data: bytes) -> str:
    """
    Same extraction as extract_text_from_pdf, but for a PDF already in memory
    (e.g. an uploaded FastAPI UploadFile's bytes) rather than a file on disk --
    used by the chat document-analysis pipeline, which never writes the upload
    to disk first.
    """
    if not data:
        raise ValueError("Empty PDF data.")

    text = ""
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text


def render_pdf_pages_to_images(file_path: Path, max_pages: int = 6, dpi: int = 150) -> List[bytes]:
    """
    Renders each page of a PDF to a PNG image.

    Used as a fallback when the text layer is missing or incomplete -- e.g. a
    portal export where lab values are drawn as canvas/SVG gauge widgets
    rather than real text, so `extract_text_from_pdf` captures the surrounding
    labels but not the actual numbers.
    """
    if not file_path.exists() or file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Invalid PDF file: {file_path}")

    images: List[bytes] = []
    with fitz.open(file_path) as doc:
        for page in doc[:max_pages]:
            pixmap = page.get_pixmap(dpi=dpi)
            images.append(pixmap.tobytes("png"))
    return images


def build_excerpt(text: str, max_chars: int = 320) -> str:
    """
    Builds a compact excerpt suitable for citation drawers and audit traces.
    """
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


# Vital/lab type keys that are ambiguous or otherwise misleading if shown to an
# LLM (or a human) as a bare snake_case key. A model reasoning over "peak_flow:
# 18 ml/s" with no other context will default to whichever meaning is most
# common in general (respiratory peak expiratory flow), even when the unit and
# document context make clear it's actually a different measurement (urology
# peak urinary flow rate / Qmax). This is the fix for that failure mode -- add
# an entry here for any key discovered to have the same problem.
_AMBIGUOUS_VITAL_LABELS: Dict[str, str] = {
    "peak_urinary_flow_rate": "Peak urinary flow rate / Qmax (urology, NOT a respiratory measurement)",
    "peak_expiratory_flow": "Peak expiratory flow (respiratory)",
    "peak_flow": "Peak flow (ambiguous historical entry -- verify respiratory vs. urology context)",
}


def vital_display_label(vtype: str) -> str:
    """
    Human-readable label for a vital/lab type key, disambiguating known
    ambiguous keys. Falls back to simple title-casing for everything else.
    """
    key = (vtype or "").strip().lower()
    if key in _AMBIGUOUS_VITAL_LABELS:
        return _AMBIGUOUS_VITAL_LABELS[key]
    return key.replace("_", " ").title()


def render_vital_for_prompt(entry: Dict, include_date: bool = True, date_prefix: str = "") -> str:
    """
    Renders a vital/lab entry as a single line for LLM-facing prompt context,
    using the disambiguated label instead of the bare type key. Use this
    everywhere a vital gets formatted into text a model will reason over.

    date_prefix: text placed before the date inside the parentheses, e.g.
    "recorded " to produce "(recorded 2026-07-07)" instead of "(2026-07-07)" --
    pass whichever matches the call site's existing wording.
    """
    label = vital_display_label(entry.get("type", ""))
    value = str(entry.get("value") or "").strip()
    unit = str(entry.get("unit") or "").strip()

    line = f"{label}: {value}"
    if unit:
        line += f" {unit}"
    if include_date:
        recorded_on = str(entry.get("recorded_on") or "").strip()
        if recorded_on:
            line += f" ({date_prefix}{recorded_on})"
    return line
