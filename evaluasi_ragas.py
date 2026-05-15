"""
Evaluasi RAG Pipeline menggunakan RAGAS
Metrik : Faithfulness · Answer Relevancy · Context Precision
Judge  : OpenAI GPT-4o mini

Alur:
  1. Jalankan setiap kasus di rag_pipeline.py
  2. Isi question, answer, context ke dataset_evaluasi.csv
  3. python evaluasi_ragas.py
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from dotenv import load_dotenv
load_dotenv()
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    LLMContextPrecisionWithoutReference,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings


# ══════════════════════════════════════════════════════════════
# KONFIGURASI
# ══════════════════════════════════════════════════════════════

# Model judge — GPT-4o mini via OpenAI API
JUDGE_MODEL     = "gpt-4o-mini"
OPENAI_API_KEY  = ""   # ← isi di sini ATAU di file .env (OPENAI_API_KEY=sk-...)

# Embedding untuk Answer Relevancy (jalan lokal, tidak butuh API)
JUDGE_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"

CSV_PATH        = "./dataset_evaluasi.csv"
OUTPUT_CSV      = "./evaluasi_ragas_hasil.csv"
CONTEXT_COLUMNS = ["context_1", "context_2", "context_3", "context_4", "context_5", "context_6", "context_7", "context_8", "context_9" , "context_10"]


# ══════════════════════════════════════════════════════════════
# BACA DATASET DARI CSV
# ══════════════════════════════════════════════════════════════

def load_dataset_from_csv(csv_path: str) -> list[dict]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File CSV tidak ditemukan: {csv_path}")

    dataset = []
    skipped = 0

    with open(csv_path, encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):
            question = row.get("question", "").strip()
            answer   = row.get("answer",   "").strip()

            if not question or not answer:
                print(f"[WARN] Baris {row_num} dilewati — 'question' atau 'answer' kosong.")
                skipped += 1
                continue

            contexts = [
                row.get(col, "").strip()
                for col in CONTEXT_COLUMNS
                if row.get(col, "").strip()
            ]

            if not contexts:
                print(f"[WARN] Baris {row_num} dilewati — tidak ada context.")
                skipped += 1
                continue

            dataset.append({
                "id_kasus" : row.get("id_kasus", str(row_num - 1)),
                "question" : question,
                "answer"   : answer,
                "contexts" : contexts,
            })

    print(f"[INFO] {len(dataset)} kasus berhasil dimuat.")
    if skipped:
        print(f"[INFO] {skipped} baris dilewati.")
    if not dataset:
        raise ValueError("Tidak ada kasus valid di CSV.")
    return dataset


# ══════════════════════════════════════════════════════════════
# SETUP JUDGE
# ══════════════════════════════════════════════════════════════

def setup_judge():
    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OpenAI API key tidak ditemukan.\n"
            "Isi OPENAI_API_KEY di bagian KONFIGURASI atau set OPENAI_API_KEY di .env"
        )

    print(f"[INFO] Judge LLM       : {JUDGE_MODEL} (OpenAI API)")
    print(f"[INFO] Judge Embedding : {JUDGE_EMBEDDING} (lokal)")

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=JUDGE_MODEL,
            api_key=api_key,
            temperature=0,
            max_tokens=2500,   # cukup untuk semua kasus RAGAS
        )
    )
    judge_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=JUDGE_EMBEDDING,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )
    return judge_llm, judge_emb


# ══════════════════════════════════════════════════════════════
# EVALUASI
# ══════════════════════════════════════════════════════════════

def run_single_metric(dataset: list[dict], metric_name: str):
    """
    Jalankan evaluasi untuk SATU metrik saja.
    metric_name: "faithfulness" | "answer_relevancy" | "context_precision"
    """
    ragas_data = [
        {
            "user_input"          : item["question"],
            "response"            : item["answer"],
            "retrieved_contexts"  : item["contexts"],
        }
        for item in dataset
    ]

    hf_dataset           = Dataset.from_list(ragas_data)
    judge_llm, judge_emb = setup_judge()

    metric_map = {
        "faithfulness"      : Faithfulness(llm=judge_llm),
        "answer_relevancy"  : AnswerRelevancy(llm=judge_llm, embeddings=judge_emb),
        "context_precision" : LLMContextPrecisionWithoutReference(llm=judge_llm),
    }

    if metric_name not in metric_map:
        raise ValueError(f"Metrik tidak dikenal: {metric_name}. Pilih: {list(metric_map.keys())}")

    run_cfg = RunConfig(
        timeout=120,      # 1 menit — cukup untuk OpenAI API
        max_retries=1,   # minimal retry — setiap retry = biaya tambahan
        max_workers=2,   # tidak terlalu paralel — hindari burst request
    )

    print("\n" + "="*65)
    print(f"  EVALUASI METRIK: {metric_name.upper()}")
    print(f"  Jumlah kasus : {len(dataset)}")
    print(f"  Judge LLM    : {JUDGE_MODEL}")
    print("="*65 + "\n")

    return evaluate(
        dataset=hf_dataset,
        metrics=[metric_map[metric_name]],
        run_config=run_cfg,
    )


# ══════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════

def plot_bar_chart(df, metric_cols, metric_labels, output_path="plot_bar_chart.png"):
    means  = [df[col].mean() for col in metric_cols if col in df.columns]
    labels = [metric_labels[col] for col in metric_cols if col in df.columns]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, means, color=colors[:len(labels)], height=0.5, edgecolor="white")

    for bar, val in zip(bars, means):
        ax.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=11, fontweight="bold"
        )

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Skor (0 – 1)", fontsize=11)
    ax.set_title("Skor Rata-Rata RAGAS per Metrik", fontsize=13, fontweight="bold", pad=12)
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)
    for ref in [0.5, 0.75]:
        ax.axvline(x=ref, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.text(ref, -0.6, str(ref), ha="center", va="top", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Disimpan → {output_path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    METRIC_COLS = [
        "faithfulness",
        "answer_relevancy",
        "llm_context_precision_without_reference",
    ]
    METRIC_LABELS = {
        "faithfulness"                            : "Faithfulness",
        "answer_relevancy"                        : "Answer Relevancy",
        "llm_context_precision_without_reference" : "Context Precision",
    }

    print(f"[INFO] Membaca dataset dari: {CSV_PATH}\n")
    dataset = load_dataset_from_csv(CSV_PATH)

    # ── PILIH METRIK YANG INGIN DIJALANKAN ──────────────────
    # Jalankan satu per satu agar tidak berat:
    #   python evaluasi_ragas.py            → jalankan semua (default)
    #   python evaluasi_ragas.py faith      → faithfulness saja
    #   python evaluasi_ragas.py relevancy  → answer relevancy saja
    #   python evaluasi_ragas.py precision  → context precision saja

    import sys
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    mode_map = {
        "faith"     : "faithfulness",
        "relevancy" : "answer_relevancy",
        "precision" : "context_precision",
        "all"       : "all",
    }
    mode = mode_map.get(arg, "all")

    import pandas as pd
    from pathlib import Path as _Path

    if mode == "all":
        # Jalankan ketiga metrik sekaligus
        metrics_to_run = ["faithfulness", "answer_relevancy", "context_precision"]
    else:
        metrics_to_run = [mode]

    all_dfs = []

    for metric_name in metrics_to_run:
        print(f"\n>>> Menjalankan: {metric_name}")
        try:
            hasil = run_single_metric(dataset, metric_name)
            df_m  = hasil.to_pandas()
            df_m.insert(0, "id_kasus", [item["id_kasus"] for item in dataset])
            all_dfs.append(df_m)

            # Simpan hasil per metrik
            out = OUTPUT_CSV.replace(".csv", f"_{metric_name}.csv")
            df_m.to_csv(out, index=False, encoding="utf-8")
            print(f"  ✅ Tersimpan → {out}")

            # Preview skor rata-rata
            col = [c for c in df_m.columns if c in METRIC_COLS]
            for c in col:
                print(f"  {METRIC_LABELS.get(c, c)}: {df_m[c].mean():.4f} (±{df_m[c].std():.4f})")

        except Exception as e:
            print(f"  ❌ Error pada {metric_name}: {e}")

    # Gabungkan semua hasil jika lebih dari satu metrik
    if len(all_dfs) > 1:
        df_final = all_dfs[0]
        for df_next in all_dfs[1:]:
            new_cols = [c for c in df_next.columns if c not in df_final.columns]
            df_final = pd.concat([df_final, df_next[new_cols]], axis=1)
    elif len(all_dfs) == 1:
        df_final = all_dfs[0]
    else:
        print("Tidak ada hasil yang berhasil.")
        exit()

    # Simpan gabungan
    df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n  ✅ Hasil gabungan disimpan → {OUTPUT_CSV}")

    # Tampilkan ringkasan
    print("\n" + "="*65)
    print("  RINGKASAN SKOR")
    print("="*65)
    print(f"  {'Metrik':<30} {'Skor':>8}  {'Std Dev':>8}")
    print(f"  {'-'*48}")
    for col in METRIC_COLS:
        if col in df_final.columns:
            print(f"  {METRIC_LABELS[col]:<30} {df_final[col].mean():>7.4f}  {df_final[col].std():>8.4f}")

    # Plot jika ada minimal satu metrik
    avail = [c for c in METRIC_COLS if c in df_final.columns]
    if avail:
        print("\n  Membuat visualisasi...")
        plot_bar_chart(df_final, avail, METRIC_LABELS)
        print(f"  ✅ Plot disimpan → plot_bar_chart.png")

    print("\n" + "="*65)
    print("  Selesai.")
    print("="*65)