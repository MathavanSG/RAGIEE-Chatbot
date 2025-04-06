import re
from typing import List, Tuple
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    """Reads and extracts all text from the PDF."""
    reader = PdfReader(pdf_path)
    full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
    return full_text


def extract_headings(text: str) -> List[Tuple[int, str]]:
    """Detects section headings (e.g., Abstract, Introduction, 1.1 Method)."""
    headings = []

    # Detect Abstract explicitly
    abstract_match = re.search(
        r"(Abstract[\s\n]*)(.*?)(?=\n\s*(?:[1I]\.|1\s|I\s|Introduction))",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if abstract_match:
        start = abstract_match.start(2)
        headings.append((start, "Abstract"))

    # General section headers: 1. Intro, A. Dataset, I. RESULTS
    heading_pattern = re.compile(
        r"^\s*((\d{1,2}(\.\d+)*|[A-Z]|[IVXLCDM]+)\.?\s+)([A-Z][^\n]{3,80})$",
        re.MULTILINE,
    )
    for match in heading_pattern.finditer(text):
        heading = match.group(4).strip()
        headings.append((match.start(), heading))

    headings.sort(key=lambda x: x[0])
    return headings


def split_text_by_headings(
    text: str, headings: List[Tuple[int, str]]
) -> List[Tuple[str, str]]:
    """Splits full text into sections based on headings."""
    sections = []
    for i, (start, heading) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        section_text = text[start:end].strip()
        sections.append((heading, section_text))
    return sections
