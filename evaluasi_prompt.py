"""
Evaluasi Few-Shot Prompt Only — Tanpa RAG
Metrik : Semantic Hit@1 · Hit@3 · MRR · Distribusi Gradasi Similarity
LLM    : Qwen2.5:3b-instruct via Ollama (lokal)
Embedding: all-MiniLM-L6-v2 (SentenceTransformer)

Tujuan:
  Mengevaluasi performa LLM dengan few-shot prompt TANPA retrieval
  sebagai baseline perbandingan terhadap sistem RAG.
  Hasil evaluasi ini dibandingkan langsung dengan evaluasi_ranking.py
  untuk mengukur kontribusi komponen RAG terhadap sistem.

Cara pakai:
  1. Pastikan Ollama berjalan dan model tersedia
  2. python evaluasi_fewshot.py
"""

import re
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer, util


# ══════════════════════════════════════════════════════════════
# KONFIGURASI
# ══════════════════════════════════════════════════════════════

LLM_MODEL_ID    = "qwen2.5:3b-instruct"
CSV_PATH        = "./dataset_evaluasi.csv"
OUTPUT_CSV      = "./evaluasi_fewshot_hasil.csv"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
THRESHOLD       = 0.75

# Threshold gradasi similarity (adaptasi Schumacher et al., 2025)
GRADASI = {
    "Exact/Near-exact" : (0.90, 1.01),
    "Highly Relevant"  : (0.75, 0.90),
    "Relevant"         : (0.60, 0.75),
    "Somewhat Related" : (0.40, 0.60),
    "Unrelated"        : (0.00, 0.40),
}


# ══════════════════════════════════════════════════════════════
# FEW-SHOT PROMPT (tanpa context RAG)
# ══════════════════════════════════════════════════════════════

FEWSHOT_PROMPT = """You are a sports medicine physician assistant specializing in running injuries. Below are examples of how to analyze a clinical presentation and generate 3 differential diagnoses.

---
EXAMPLE 1:

Clinical presentation:
Male patient, 28 years old, competitive runner (60 km/week). Reports sharp, burning pain on the outer right knee, consistently appearing after 6-7 km of running. Pain worsens going downhill and subsides with rest. No swelling, no locking, no direct trauma history. Physical exam shows tenderness at the lateral femoral epicondyle.

Based on the provided clinical presentation and context, here are three differential diagnoses:

1. **Iliotibial Band Syndrome (ITBS)**
   Supporting findings: Sharp burning lateral knee pain appearing consistently after 6-7 km; tenderness at the lateral femoral epicondyle; pain aggravated by downhill running and relieved by rest.
   Against: No joint effusion, no locking, no acute trauma — findings that would suggest intra-articular pathology.
   Clinical reasoning: The combination of distance-triggered lateral knee pain, epicondyle tenderness, and downhill aggravation is the hallmark presentation of ITBS in high-mileage runners.

2. **Lateral Meniscus Tear**
   Supporting findings: Lateral knee pain in a runner; tenderness near the lateral joint line.
   Against: Absence of joint locking, swelling, and acute trauma history makes significant meniscal injury less likely.
   Clinical reasoning: Meniscal tears typically present with mechanical symptoms such as locking or effusion; their absence makes this a lower-priority differential.

3. **Biceps Femoris Tendinopathy**
   Supporting findings: Lateral knee pain in a high-mileage competitive runner; gradual onset consistent with overuse.
   Against: Tenderness localised at the lateral femoral epicondyle rather than the fibular head insertion site of the biceps femoris tendon.
   Clinical reasoning: Biceps femoris tendinopathy is plausible in high-mileage runners but the epicondyle tenderness location favours ITBS over tendon pathology.


EXAMPLE 2:

Clinical presentation:
Female patient, 42 years old, recreational runner (25 km/week). Sharp pain at the bottom of the left heel, worst on first steps in the morning. Improves after walking 5-10 minutes. Bilateral tight calf muscles. No numbness or tingling.

Based on the provided clinical presentation and context, here are three differential diagnoses:

1. **Plantar Fasciitis**
   Supporting findings: Sharp plantar heel pain worst with first steps in the morning; improves after walking 5-10 minutes; bilateral tight calf muscles as a recognised risk factor.
   Against: No numbness or tingling, ruling out neurogenic causes of heel pain.
   Clinical reasoning: The classic first-step morning pain that improves with activity, combined with tight calf muscles, is the defining presentation of plantar fasciitis.

2. **Calcaneal Stress Fracture**
   Supporting findings: Heel pain in a female recreational runner; gradual onset consistent with repetitive loading.
   Against: The warm-up effect — pain improving with continued activity — is more typical of plantar fasciitis than stress fracture, which usually worsens with loading.
   Clinical reasoning: Calcaneal stress fracture warrants consideration in female runners, but the activity-related improvement pattern is more consistent with plantar fasciitis.

3. **Tarsal Tunnel Syndrome**
   Supporting findings: Heel and plantar foot pain in a runner.
   Against: Absence of numbness, tingling, or radiating pain argues strongly against tarsal tunnel syndrome, which characteristically involves neurological symptoms.
   Clinical reasoning: Tarsal tunnel syndrome is unlikely without neurological symptoms; it is included to complete the differential for plantar heel pain.




Clinical presentation:
{question}

Top 3 Differential Diagnosis:
"""

# ══════════════════════════════════════════════════════════════
# BACA DATASET
# ══════════════════════════════════════════════════════════════

def load_dataset(csv_path: str) -> list[dict]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {csv_path}")

    dataset = []
    with open(csv_path, encoding="utf-8-sig", errors="replace", newline="") as f:
        for row_num, row in enumerate(csv.DictReader(f), start=2):
            question     = row.get("question",     "").strip()
            ground_truth = row.get("ground_truth", "").strip()
            if not question or not ground_truth:
                print(f"[WARN] Baris {row_num} dilewati — kolom wajib kosong.")
                continue
            dataset.append({
                "id_kasus"    : row.get("id_kasus", str(row_num - 1)),
                "question"    : question,
                "ground_truth": ground_truth,
            })

    print(f"[INFO] {len(dataset)} kasus dimuat.")
    return dataset


# ══════════════════════════════════════════════════════════════
# EKSTRAKSI SEMUA NAMA DIAGNOSIS (Top-1 s/d Top-3)
# ══════════════════════════════════════════════════════════════

def ekstrak_semua_diagnosis(teks: str) -> list[str]:
    marker      = re.search(r'Differential Diagnosis\s*:\s*', teks, re.IGNORECASE)
    search_text = teks[marker.end():] if marker else teks

    names = []

    # Pola 1: "1. **Nama**" atau "1. Nama"
    for m in re.finditer(
        r"^[ \t]*\d+[\.)][\.)\s]{0,3}\*{0,2}"
        r"([A-Z][A-Za-z\(\)\-\/][A-Za-z\(\)\-\/\.' ]{2,58}?)"
        r"\*{0,2}[ \t]*$",
        search_text, re.MULTILINE
    ):
        raw  = m.group(1).strip().rstrip(':').strip()
        name = re.split(r'  +|\t', raw)[0].strip()
        if name and len(name) > 4 and name not in names:
            names.append(name)

    # Pola 2: **Bold Name** (fallback)
    if len(names) < 3:
        for m in re.finditer(
            r"\*\*([A-Z][A-Za-z\(\)\-\/\.' ]{5,60}?)\*\*", search_text
        ):
            name = m.group(1).strip()
            if name and name not in names:
                names.append(name)

    return names[:3]


# ══════════════════════════════════════════════════════════════
# COSINE SIMILARITY & GRADASI
# ══════════════════════════════════════════════════════════════

def cosine_sim(emb, text_a: str, text_b: str) -> float:
    va = emb.encode(text_a, convert_to_tensor=True)
    vb = emb.encode(text_b, convert_to_tensor=True)
    return util.cos_sim(va, vb).item()


def tentukan_gradasi(skor: float) -> str:
    for label, (low, high) in GRADASI.items():
        if low <= skor < high:
            return label
    return "Unrelated"


def cari_rank(emb, ground_truth: str, diagnoses: list[str]) -> tuple[int, list[float]]:
    scores = []
    for diag in diagnoses:
        if diag == "EXTRACTION_FAILED":
            scores.append(0.0)
        else:
            scores.append(cosine_sim(emb, ground_truth, diag))

    rank = 0
    for i, score in enumerate(scores):
        if score >= THRESHOLD:
            rank = i + 1
            break

    return rank, scores


# ══════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════

def plot_hasil(hasil_rows: list[dict], total: int, output_path: str):
    hit1 = sum(1 for r in hasil_rows if r["hit_at_1"] == "HIT")
    hit3 = sum(1 for r in hasil_rows if r["hit_at_3"] == "HIT")
    mrr  = sum(r["reciprocal_rank"] for r in hasil_rows) / total

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    # Plot 1: Hit@1 vs Hit@3
    categories = ["Hit@1", "Hit@3"]
    values     = [hit1 / total * 100, hit3 / total * 100]
    bars = axes[0].bar(categories, values, color=["#42A5F5", "#66BB6A"],
                       width=0.4, edgecolor="white", zorder=3)
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 1.5,
                     f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, 115)
    axes[0].set_ylabel("Akurasi (%)", fontsize=11)
    axes[0].set_title("Hit@1 vs Hit@3\n(Few-Shot Only)", fontsize=12, fontweight="bold")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # Plot 2: Distribusi Rank
    rank_counts = {1: 0, 2: 0, 3: 0, 0: 0}
    for r in hasil_rows:
        rank_counts[r["rank"]] = rank_counts.get(r["rank"], 0) + 1
    rank_labels = ["Rank 1", "Rank 2", "Rank 3", "Tidak\nDitemukan"]
    rank_values = [rank_counts[1], rank_counts[2], rank_counts[3], rank_counts[0]]
    bars2 = axes[1].bar(rank_labels, rank_values,
                        color=["#66BB6A", "#FFA726", "#EF5350", "#BDBDBD"],
                        width=0.5, edgecolor="white", zorder=3)
    for bar, val in zip(bars2, rank_values):
        if val > 0:
            axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.1,
                         str(val), ha="center", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0, max(rank_values) + 3)
    axes[1].set_ylabel("Jumlah Kasus", fontsize=11)
    axes[1].set_title("Distribusi Posisi\nGround Truth", fontsize=12, fontweight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # Plot 3: MRR
    axes[2].barh(["Few-Shot\nOnly"], [mrr], color="#AB47BC",
                 height=0.4, edgecolor="white", zorder=3)
    axes[2].text(mrr + 0.01, 0, f"{mrr:.4f}",
                 va="center", ha="left", fontsize=13, fontweight="bold")
    axes[2].set_xlim(0, 1.15)
    axes[2].set_xlabel("MRR Score (0 – 1)", fontsize=11)
    axes[2].set_title("Mean Reciprocal Rank\n(MRR)", fontsize=12, fontweight="bold")
    for ref in [0.33, 0.5, 0.75, 1.0]:
        axes[2].axvline(x=ref, color="gray", linewidth=0.7,
                        linestyle="--", alpha=0.5)
        axes[2].text(ref, -0.45, str(ref), ha="center",
                     va="top", fontsize=8, color="gray")
    axes[2].spines[["top", "right"]].set_visible(False)
    axes[2].grid(axis="x", linestyle="--", alpha=0.3, zorder=0)

    # Plot 4: Distribusi Gradasi
    gradasi_labels  = list(GRADASI.keys())
    gradasi_counts  = [sum(1 for r in hasil_rows if r["gradasi_similarity"] == l)
                       for l in gradasi_labels]
    gradasi_pct     = [c / total * 100 for c in gradasi_counts]
    gradasi_colors  = ["#1B5E20", "#66BB6A", "#FFA726", "#FF7043", "#B71C1C"]
    bars3 = axes[3].barh(gradasi_labels[::-1], gradasi_pct[::-1],
                         color=gradasi_colors[::-1], height=0.5,
                         edgecolor="white", zorder=3)
    for bar, val, count in zip(bars3, gradasi_pct[::-1], gradasi_counts[::-1]):
        if val > 0:
            axes[3].text(val + 0.5, bar.get_y() + bar.get_height()/2,
                         f"{count} ({val:.1f}%)",
                         va="center", ha="left", fontsize=10, fontweight="bold")
    axes[3].set_xlim(0, 115)
    axes[3].set_xlabel("Persentase Kasus (%)", fontsize=11)
    axes[3].set_title("Distribusi Gradasi\nSimilarity", fontsize=12, fontweight="bold")
    axes[3].spines[["top", "right"]].set_visible(False)
    axes[3].grid(axis="x", linestyle="--", alpha=0.3, zorder=0)
    axes[3].tick_params(axis="y", labelsize=9)

    plt.suptitle(
        f"Evaluasi Few-Shot Only (Tanpa RAG)  |  n={total}  |  threshold={THRESHOLD}  |  {LLM_MODEL_ID}",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] disimpan → {output_path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # 1. Muat dataset
    print(f"[INFO] Membaca dataset dari: {CSV_PATH}")
    dataset = load_dataset(CSV_PATH)

    # 2. Muat embedding model
    print(f"[INFO] Memuat embedding: {EMBEDDING_MODEL}")
    emb = SentenceTransformer(EMBEDDING_MODEL)

    # 3. Inisialisasi LLM
    print(f"[INFO] Menghubungkan ke Ollama: {LLM_MODEL_ID}")
    llm = ChatOllama(model=LLM_MODEL_ID, temperature=0.3)

    print(f"\n{'='*72}")
    print(f"  EVALUASI FEW-SHOT ONLY (TANPA RAG)")
    print(f"  Metrik    : Hit@1 · Hit@3 · MRR · Gradasi Similarity")
    print(f"  Threshold : {THRESHOLD}")
    print(f"  LLM       : {LLM_MODEL_ID}")
    print(f"  Embedding : {EMBEDDING_MODEL}")
    print(f"{'='*72}")

    hasil_rows    = []
    total         = len(dataset)
    sum_rr        = 0.0
    gagal_ekstrak = 0

    for kasus in dataset:
        print(f"  Kasus {kasus['id_kasus']:>3}...", end=" ", flush=True)

        try:
            # Generate output dari LLM dengan few-shot prompt
            prompt_text = FEWSHOT_PROMPT.format(question=kasus["question"])
            response    = llm.invoke(prompt_text)
            teks_llm    = response.content

            # Ekstrak semua diagnosis
            diagnoses = ekstrak_semua_diagnosis(teks_llm)
            if not diagnoses:
                gagal_ekstrak += 1
                diagnoses = ["EXTRACTION_FAILED"]

            # Hitung rank dan skor
            rank, scores = cari_rank(emb, kasus["ground_truth"], diagnoses)

            rr   = 1.0 / rank if rank > 0 else 0.0
            sum_rr += rr

            hit1 = "HIT" if rank == 1 else "MISS"
            hit3 = "HIT" if rank in (1, 2, 3) else "MISS"

            skor_max = max(scores) if scores else 0.0
            gradasi  = tentukan_gradasi(skor_max)

            print(
                f"Rank={rank if rank > 0 else '-'} | "
                f"RR={rr:.3f} | Hit@1={hit1} | Hit@3={hit3} | "
                f"{gradasi} | GT: {kasus['ground_truth'][:25]}"
            )

            hasil_rows.append({
                "id_kasus"          : kasus["id_kasus"],
                "question"          : kasus["question"][:80],
                "ground_truth"      : kasus["ground_truth"],
                "diagnosis_1"       : diagnoses[0] if len(diagnoses) > 0 else "",
                "diagnosis_2"       : diagnoses[1] if len(diagnoses) > 1 else "",
                "diagnosis_3"       : diagnoses[2] if len(diagnoses) > 2 else "",
                "skor_1"            : round(scores[0] * 100, 2) if len(scores) > 0 else 0,
                "skor_2"            : round(scores[1] * 100, 2) if len(scores) > 1 else 0,
                "skor_3"            : round(scores[2] * 100, 2) if len(scores) > 2 else 0,
                "skor_tertinggi"    : round(skor_max * 100, 2),
                "rank"              : rank,
                "reciprocal_rank"   : round(rr, 4),
                "hit_at_1"          : hit1,
                "hit_at_3"          : hit3,
                "gradasi_similarity": gradasi,
                "output_llm"        : teks_llm[:300].replace("\n", " "),
            })

        except Exception as e:
            print(f"ERROR: {e}")
            hasil_rows.append({
                "id_kasus"          : kasus["id_kasus"],
                "question"          : kasus["question"][:80],
                "ground_truth"      : kasus["ground_truth"],
                "diagnosis_1"       : "ERROR",
                "diagnosis_2"       : "", "diagnosis_3"       : "",
                "skor_1"            : 0, "skor_2"            : 0,
                "skor_3"            : 0, "skor_tertinggi"    : 0,
                "rank"              : 0, "reciprocal_rank"   : 0.0,
                "hit_at_1"          : "ERROR", "hit_at_3"    : "ERROR",
                "gradasi_similarity": "Unrelated",
                "output_llm"        : str(e)[:200],
            })

    # ── Ringkasan ──────────────────────────────────────────────
    mrr      = sum_rr / total
    hit1_n   = sum(1 for r in hasil_rows if r["hit_at_1"] == "HIT")
    hit3_n   = sum(1 for r in hasil_rows if r["hit_at_3"] == "HIT")
    hit1_pct = hit1_n / total * 100
    hit3_pct = hit3_n / total * 100

    rank_dist = {1: 0, 2: 0, 3: 0, 0: 0}
    for r in hasil_rows:
        rank_dist[r["rank"]] = rank_dist.get(r["rank"], 0) + 1

    gradasi_dist = {label: 0 for label in GRADASI}
    for r in hasil_rows:
        gradasi_dist[r["gradasi_similarity"]] += 1

    print(f"\n{'='*72}")
    print(f"  RINGKASAN HASIL — FEW-SHOT ONLY (TANPA RAG)")
    print(f"{'='*72}")
    print(f"  Hit@1  : {hit1_pct:.2f}%  ({hit1_n}/{total})")
    print(f"  Hit@3  : {hit3_pct:.2f}%  ({hit3_n}/{total})")
    print(f"  MRR    : {mrr:.4f}")
    print(f"{'─'*50}")
    print(f"  Distribusi rank ground truth:")
    print(f"    Rank 1          : {rank_dist[1]:>3} kasus")
    print(f"    Rank 2          : {rank_dist[2]:>3} kasus")
    print(f"    Rank 3          : {rank_dist[3]:>3} kasus")
    print(f"    Tidak ditemukan : {rank_dist[0]:>3} kasus")
    print(f"{'─'*50}")
    print(f"  Distribusi gradasi similarity:")
    for label, count in gradasi_dist.items():
        low, high = GRADASI[label]
        pct = count / total * 100
        bar = "█" * int(pct / 5)
        print(f"    {label:<22} : {count:>3} ({pct:>5.1f}%)  {bar}")
    if gagal_ekstrak:
        print(f"\n  Gagal ekstraksi : {gagal_ekstrak} kasus")
    print(f"  Threshold       : {THRESHOLD}")
    print(f"  LLM             : {LLM_MODEL_ID}")
    print(f"{'='*72}")

    # ── Simpan CSV ─────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(hasil_rows[0].keys()))
        writer.writeheader()
        writer.writerows(hasil_rows)
    print(f"\n  ✅ Hasil disimpan → {OUTPUT_CSV}")

    # ── Plot ───────────────────────────────────────────────────
    plot_hasil(hasil_rows, total, output_path="plot_fewshot.png")
    print(f"  ✅ Plot disimpan → plot_fewshot.png")