import re
import pymupdf as fitz


def load_pages(pdf_path: str) -> dict[int, str]:
    """
    Mengekstrak teks per halaman dari PDF.
    Jika ada tabel, tabel diekstrak ke dalam format markdown.
    Mengembalikan dict dengan key nomor halaman (1-indexed) dan value teks halaman.
    """
    doc = fitz.open(pdf_path)
    pages = {}

    for page_num, page in enumerate(doc, start=1):
        # Coba ekstrak tabel sebagai markdown jika didukung versi PyMuPDF
        table_markdowns = []
        try:
            tabs = page.find_tables()
            for tab in tabs:
                table_md = tab.to_markdown().replace("\u200b", " ").replace("\u00ad", " ")
                table_markdowns.append(re.sub(r"[ \t]+", " ", table_md))
        except Exception:
            pass

        raw_text = page.get_text().replace("\u200b", " ").replace("\u00ad", " ")
        text = re.sub(r"[ \t]+", " ", raw_text).strip()
        if table_markdowns:
            text += "\n\n" + "\n\n".join(table_markdowns)

        pages[page_num] = text

    return pages
