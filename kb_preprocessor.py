"""
KB Preprocessor — Membersihkan noise struktural dari dokumen PDF akademik
sebelum dimasukkan ke pipeline RAG.

Cara integrasi di rag_pipeline.py:
    from kb_preprocessor import preprocess_documents, filter_chunks
    raw_docs = preprocess_documents(raw_docs)     # sebelum split
    chunks   = splitter.split_documents(raw_docs)
    chunks   = filter_chunks(chunks)              # setelah split
"""

import re
import sys
from typing import List
from langchain_core.documents import Document


def _log(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()


def clean_text(text: str) -> str:

    # 1. Hapus section daftar pustaka
    text = re.sub(
        r'\n\s*(References|REFERENCES|Bibliography|BIBLIOGRAPHY|'
        r'Daftar Pustaka|DAFTAR PUSTAKA)\s*\n.*',
        '\n', text, flags=re.DOTALL
    )

    # 2. Perbaiki hyphenation PDF dua kolom: "syn-\ndrome" -> "syndrome"
    text = re.sub(r'-\n\s*', '', text)

    # 2b. Hapus karakter kontrol PDF: BEL \x07, form feed \x0c, dll
    text = re.sub(r'[\x00-\x08\x0b\x0e-\x1f\x7f]', '', text)

    # 2c. Hapus URL dx.doi.org dan http doi.org yang sering muncul dalam referensi PDF
    text = re.sub(r'dx\.doi\.org/\S+', '', text)
    text = re.sub(r'https?://doi\.org/\S+', '', text)

    # 3. Hapus sitasi inline: [1], [1-4], [1, 2, 3]
    text = re.sub(r'\[\d+(?:[,\s\-]\d+)*\]', '', text)
    # Superscript angka setelah kata: "runners12" -> "runners"
    text = re.sub(r'(?<=[a-zA-Z])\d{1,3}(?=[,\.\s])', '', text)

    # 3b. Hapus baris referensi jurnal format "Nama Inisial. Tahun. Judul. Journal"
    text = re.sub(
        r'^[A-Z][a-z]+\s+[A-Z]{1,3}[,\s]+.{10,200}\d{4}[^\n]{0,100}$',
        '', text, flags=re.MULTILINE
    )
    # Hapus baris DOI standalone
    text = re.sub(r'^DOI\s+10\.\S+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d{2,4}\(\d+\):\d+[–\-]\d+$', '', text, flags=re.MULTILINE)

    # 3c. Hapus blok referensi bernomor yang lolos — format "(angka):hal\nAngka.\nNama..."
    # Contoh: "(11):A1-33.\n62.\nThomas JL..."
    text = re.sub(
        r'^\(\d+\):[A-Za-z0-9\-]+\.?\s*\n.*',
        '', text, flags=re.MULTILINE | re.DOTALL
    )
    # Hapus referensi format "Kota: Penerbit, tahun: chapter angka, hal\nAngka. Nama..."
    text = re.sub(
        r'^[A-Z][a-z]+.*\d{4}:\s*chapter\s*\d+.*\n\d+\.\s+[A-Z].*',
        '', text, flags=re.MULTILINE
    )
    # Hapus baris referensi dalam chunk: "\n62.\nThomas JL, ..."
    text = re.sub(
        r'\n\d{1,3}\.\s*\n([A-Z][a-z]+\s+[A-Z]{1,3}[,\s].{20,200}\d{4}[^\n]{0,80})',
        '', text, flags=re.MULTILINE
    )

    # 4. Hapus baris metadata jurnal/halaman
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append('')
            continue
        # Nomor halaman standalone
        if re.match(r'^\d{1,4}$', stripped):
            continue
        # Copyright, DOI, Vol, Issue
        if re.match(r'^(©|Vol\.|Issue|DOI:|https?://)', stripped, re.IGNORECASE):
            continue
        # Metadata jurnal: "J Sports Med 2006;36(3):199"
        if re.match(r'^[A-Z][A-Za-z\s&\.]+\d{4}\s*[;:]\s*\d+[\s\(\d]', stripped):
            continue
        # Label artikel
        if re.match(
            r'^(REVIEW ARTICLE|ORIGINAL ARTICLE|CASE REPORT|SYSTEMATIC REVIEW|'
            r'NARRATIVE REVIEW|CLINICAL REVIEW|SHORT REPORT|LETTER)$',
            stripped, re.IGNORECASE
        ):
            continue
        # Kode lisensi: "$39.95/0"
        if re.match(r'^\d+\.\d+/\d+$', stripped):
            continue
        # Copyright panjang
        if re.match(r'^©\s*\d{4}.{5,80}(reserved|rights)', stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    # 5. Hapus blok tabel systematic review
    text = re.sub(
        r'[A-Z][a-z]+\s+et al\.\s*\(\d{4}\)\s*\n'
        r'(Retrospective|Prospective|Cross.sectional|Cohort|Case.control|RCT)'
        r'.{0,500}?(?=\n\n|\Z)',
        '', text, flags=re.DOTALL | re.IGNORECASE
    )

    # 6. Hapus referensi bernomor
    text = re.sub(
        r'^\s*\d{1,3}\.\s+[A-Z][a-z]+.{10,200}\d{4}[^\n]{0,60}$',
        '', text, flags=re.MULTILINE
    )

    # 7. Gabungkan baris lanjutan paragraf
    text = re.sub(r'(?<=[a-zA-Z,;])\n(?=[a-z])', ' ', text)

    # 8. Bersihkan spasi dan baris kosong berlebihan
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# Kata yang menandakan chunk dimulai di tengah kalimat
_STOPWORDS_START = {
    'to', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'by', 'for', 'with',
    'and', 'or', 'but', 'as', 'is', 'are', 'was', 'were', 'it', 'this',
    'that', 'these', 'those', 'its', 'their', 'they', 'from', 'into',
}


def is_valid_chunk(text: str, min_words: int = 15) -> bool:

    words = text.split()

    # Filter 1: terlalu pendek
    if len(words) < min_words:
        return False

    # Filter 2: rasio huruf terlalu rendah (tabel angka)
    alpha_count = sum(1 for c in text if c.isalpha())
    if len(text) > 0 and alpha_count / len(text) < 0.45:
        return False

    # Filter 3: mayoritas isinya author et al.
    author_pattern = re.findall(
        r'[A-Z][a-z]+(?:\s+[A-Z][A-Za-z]+)*\s+et al\.', text
    )
    if len(author_pattern) > 2 and len(words) < 50:
        return False

    # Filter 4: chunk metadata jurnal (kode lisensi ISSN)
    if re.search(r'\d{4}-\d{4}/\d{2}/', text):
        return False

    # Filter 5: chunk berisi DOI
    if re.search(r'\bDOI\s*10\.\d{4}', text, re.IGNORECASE):
        return False
    if re.search(r'https?://doi\.org/', text):
        return False

    # Filter 6: referensi jurnal — Author inisial + tahun + journal + halaman
    if re.search(r'British Journal|Sports Medicine|Orthopaedics|Physical Therapy', text) \
       and re.search(r'\d+\(\d+\):\d+', text):
        return False

    # Filter 7: kalimat diakhiri "et al." tanpa kalimat lengkap
    if re.search(r'et al\.?\s*$', text.strip()) and len(words) < 60:
        return False

    # Filter 8: referensi bernomor pendek
    if re.match(r'^\d{1,3}\.\s+[A-Z][a-z]+', text.strip()) and len(words) < 40:
        return False

    # Filter 9: header paper — nama author dengan inisial tanpa verba klinis
    has_author_initials = bool(re.search(
        r'[A-Z][a-z]+\s+[A-Z]\.(?:[A-Z]\.)?\s+[A-Z][a-z]+', text
    ))
    has_clinical_verb = bool(re.search(
        r'\b(presents?|is\s+a|occurs?|aggravat|complains?|locali[sz]ed|'
        r'associated|charact|defined|described|known\s+as|noted|found|'
        r'observed|reported|classified|divided|caused|resulting|due\s+to)\b',
        text, re.IGNORECASE
    ))
    if has_author_initials and not has_clinical_verb:
        return False

    # Filter 10: chunk tidak koheren akibat layout dua kolom PDF
    lines_non_empty = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines_non_empty) >= 4:
        short_lines = sum(1 for l in lines_non_empty if len(l.split()) <= 5)
        if short_lines / len(lines_non_empty) > 0.65:
            return False

    # Filter 11: chunk terpotong — dimulai dengan stopword
    first_word = text.strip().split()[0].lower() if text.strip() else ''
    if first_word in _STOPWORDS_START:
        return False

    # Filter 12: chunk diakhiri tiba-tiba dengan kata + koma
    if re.search(
        r'(management|treatment|load|activity|training|exercise|therapy),$',
        text.strip(), re.IGNORECASE
    ):
        return False

    # Filter 13: chunk berakhir dengan nama orang — ciri teks terpotong dua kolom
    if re.search(r',\s+[A-Z][a-z]+\s+[A-Z][a-z]+\.\s*$', text.strip()):
        return False

    # Filter 14: chunk berisi referensi bernomor dalam badan teks
    # Contoh: "(11):A1-33.\n62.\nThomas JL, Christensen JC..."
    if re.search(r'\(\d+\):[A-Za-z0-9\-]+', text):
        return False
    if re.search(r'\n\d{1,3}\.\s*\n[A-Z][a-z]+\s+[A-Z]{1,3}', text):
        return False

    # Filter 15: chunk mengandung dx.doi.org atau karakter kontrol PDF
    if 'dx.doi.org' in text or re.search(r'[\x00-\x08\x0b\x0e-\x1f]', text):
        return False

    # Filter 16: chunk mayoritas referensi jurnal bernomor (>= 2 entri)
    ref_entries = re.findall(r'\d{1,3}\.\s+[A-Z][a-z]+\s+[A-Z]{1,3}', text)
    if len(ref_entries) >= 2:
        return False

    return True


def preprocess_documents(docs: List[Document]) -> List[Document]:
    _log(f"[PREPROCESS] Memulai preprocessing {len(docs)} dokumen...")
    cleaned_docs = []
    skipped = 0
    for doc in docs:
        cleaned_text = clean_text(doc.page_content)
        if len(cleaned_text.split()) < 30:
            skipped += 1
            continue
        cleaned_docs.append(Document(
            page_content=cleaned_text,
            metadata=doc.metadata,
        ))
    _log(f"[PREPROCESS] Selesai: {len(docs)} dok -> {len(cleaned_docs)} bersih "
         f"({skipped} dilewati karena terlalu pendek setelah cleaning)")
    return cleaned_docs


def filter_chunks(chunks: List[Document]) -> List[Document]:
    _log(f"[FILTER] Memulai filter {len(chunks)} chunks...")
    valid   = [c for c in chunks if is_valid_chunk(c.page_content)]
    removed = len(chunks) - len(valid)
    _log(f"[FILTER] Selesai: {len(chunks)} chunk -> {len(valid)} valid "
         f"({removed} chunk dibuang karena terlalu pendek/noise)")
    return valid