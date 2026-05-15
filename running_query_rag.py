"""
Collect Pipeline Output
Membaca query dari dataset_evaluasi.csv, menjalankan RAG pipeline
untuk setiap kasus, lalu mengisi kolom answer dan context_1..5 secara otomatis.

Cara pakai:
  1. Isi kolom 'question' dan 'ground_truth' di dataset_evaluasi.csv
  2. Kosongkan kolom 'answer' dan 'context_1..5'
  3. python collect_pipeline_output.py
  4. Kolom answer dan context terisi otomatis
"""

import csv
from pathlib import Path
from rag_pipeline import RAGDiagnosisPipeline

# ══════════════════════════════════════════════════════════════
# KONFIGURASI
# ══════════════════════════════════════════════════════════════

INPUT_CSV       = "./dataset_evaluasi.csv"
OUTPUT_CSV      = "./dataset_evaluasi.csv"   # overwrite file yang sama
CONTEXT_COLS    = ["context_1", "context_2", "context_3", "context_4", "context_5", "context_6", "context_7", "context_8", "context_9" , "context_10"]

# Set True untuk skip kasus yang sudah ada jawabannya
# Set False untuk overwrite semua kasus
SKIP_IF_FILLED  = False


# ══════════════════════════════════════════════════════════════
# BACA CSV
# ══════════════════════════════════════════════════════════════

def load_csv(csv_path: str) -> tuple[list[dict], list[str]]:
    """Baca CSV, kembalikan (rows, fieldnames)."""
    with open(csv_path, encoding="utf-8-sig", errors="replace", newline="") as f:
        reader    = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows      = list(reader)
    return rows, fieldnames


def save_csv(csv_path: str, rows: list[dict], fieldnames: list[str]):
    """Simpan rows ke CSV."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # 1. Baca CSV
    print(f"[INFO] Membaca: {INPUT_CSV}")
    rows, fieldnames = load_csv(INPUT_CSV)
    print(f"[INFO] {len(rows)} kasus ditemukan.")

    # Pastikan kolom yang dibutuhkan ada
    required = ["id_kasus", "question", "ground_truth", "answer"] + CONTEXT_COLS
    for col in required:
        if col not in fieldnames:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di CSV. Kolom tersedia: {fieldnames}")

    # 2. Tentukan kasus mana yang perlu diproses
    to_process = []
    to_skip    = []
    for row in rows:
        already_filled = bool(row.get("answer", "").strip())
        if SKIP_IF_FILLED and already_filled:
            to_skip.append(row)
        else:
            to_process.append(row)

    print(f"[INFO] Akan diproses : {len(to_process)} kasus")
    if to_skip:
        print(f"[INFO] Di-skip (sudah ada answer) : {len(to_skip)} kasus")
        print(f"[INFO] Set SKIP_IF_FILLED = False untuk overwrite semua.")

    if not to_process:
        print("[INFO] Semua kasus sudah terisi. Tidak ada yang perlu diproses.")
        exit()

    # 3. Inisialisasi pipeline — hanya sekali
    print("\n[INFO] Inisialisasi RAG pipeline...")
    pipeline = RAGDiagnosisPipeline()
    # force_rebuild_index=True agar preprocessing dijalankan
    # dan vector store dibangun ulang dari KB terbaru.
    # Ganti ke False setelah run pertama untuk menghemat waktu.
    pipeline.initialize(force_rebuild_index=True)
    print("[INFO] Pipeline siap.\n")

    # 4. Proses setiap kasus
    success = 0
    failed  = 0

    for row in to_process:
        id_kasus = row.get("id_kasus", "?")
        question = row.get("question", "").strip()

        if not question:
            print(f"[WARN] Kasus {id_kasus}: kolom 'question' kosong, dilewati.")
            failed += 1
            continue

        print(f"[INFO] Kasus {id_kasus}: {question[:70]}...")

        try:
            result = pipeline.query(question)

            # Isi kolom answer
            row["answer"] = result["diagnosis_text"]

            # Isi kolom context_1..5
            kb_docs = result.get("kb_docs", [])
            for i, col in enumerate(CONTEXT_COLS):
                if i < len(kb_docs):
                    row[col] = kb_docs[i]["content"]
                else:
                    row[col] = ""   # kosong jika kurang dari 5 chunk

            print(f"  ✓ Selesai — {len(kb_docs)} context diambil")
            success += 1

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            row["answer"] = f"ERROR: {e}"
            for col in CONTEXT_COLS:
                row[col] = ""
            failed += 1

        # Simpan setelah setiap kasus — aman dari crash di tengah jalan
        save_csv(OUTPUT_CSV, rows, fieldnames)

    # 5. Ringkasan
    print(f"\n{'='*65}")
    print(f"  Selesai.")
    print(f"  Berhasil : {success} kasus")
    print(f"  Gagal    : {failed} kasus")
    print(f"  Di-skip  : {len(to_skip)} kasus")
    print(f"  Output   : {OUTPUT_CSV}")
    print(f"{'='*65}")