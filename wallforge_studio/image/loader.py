"""Carga de imágenes y PDFs para WallForge Studio."""
from __future__ import annotations

from pathlib import Path
import numpy as np


_SUPPORTED_IMG = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_image(path: Path, pdf_dpi: int = 200) -> np.ndarray:
    """
    Carga una imagen o PDF como array BGR (OpenCV).
    PDF: primera página a pdf_dpi ppp.
    Lanza RuntimeError si el formato no está soportado o el archivo no existe.
    """
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Archivo no encontrado: {path}")

    suffix = path.suffix.lower()

    if suffix in _SUPPORTED_IMG:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"No se pudo abrir la imagen: {path}")
        return img

    if suffix == ".pdf":
        return _load_pdf(path, pdf_dpi)

    raise RuntimeError(
        f"Formato '{suffix}' no soportado.\n"
        f"Formatos válidos: {', '.join(sorted(_SUPPORTED_IMG))} y .pdf"
    )


def _load_pdf(path: Path, dpi: int) -> np.ndarray:
    import cv2

    # Intento 1: pymupdf (más ligero, no necesita poppler)
    try:
        import pymupdf  # noqa: F401 – nombre moderno
        _mod = __import__("pymupdf")
    except ImportError:
        try:
            import fitz as _mod  # type: ignore
        except ImportError:
            _mod = None

    if _mod is not None:
        doc = _mod.open(str(path))
        page = doc[0]
        mat = _mod.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, colorspace=_mod.csRGB)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # Intento 2: pdf2image + poppler
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(str(path), dpi=dpi, first_page=1, last_page=1)
        if not pages:
            raise RuntimeError("El PDF no tiene páginas.")
        import numpy as _np
        return cv2.cvtColor(_np.array(pages[0]), cv2.COLOR_RGB2BGR)
    except ImportError:
        pass

    raise RuntimeError(
        "Para cargar PDFs instala una de estas opciones:\n"
        "  pip install pymupdf\n"
        "  pip install pdf2image  +  sudo apt install poppler-utils"
    )
