"""
RAG Pipeline — Diagnosis Banding Cedera Olahraga Lari
dengan ICD-10 Validator

Arsitektur:
  1. Query → Retrieval KB (.txt) → Konteks medis
  2. Konteks + Query → LLM → Teks diagnosis (nama kondisi)
  3. Nama kondisi → Lookup FAISS ICD-10 → Kode ICD-10 deterministik
  4. Output final: Diagnosis + Kode ICD-10 tervalidasi

Stack: LangChain · FAISS · all-MiniLM-L6-v2 (HuggingFace) · Ollama (local LLM) · PyMuPDF

CHANGELOG v2:
  - CHUNK_SIZE naik 200 → 800, CHUNK_OVERLAP 0 → 150
    → Setiap chunk kini merepresentasikan satu topik medis utuh
  - TOP_K_DOCS naik 3 → 7
    → Lebih banyak konteks relevan masuk ke LLM
  - Separators diurutkan dari header markdown dulu
    → Chunking mengikuti batas section, bukan batas karakter sembarangan
  - temperature turun 0.3 → 0.1
    → Output LLM lebih konsisten dan deterministik
  - Prompt diperbarui: tidak lagi "ONLY on context" tapi "prioritize context"
    → Menghindari hallucination akibat konteks fragmentaris
  - FORCE_REBUILD_INDEX default True saat pertama run setelah ubah KB/chunking
"""

import re
from pathlib import Path
from typing import List, Dict, Any
from kb_preprocessor import preprocess_documents, filter_chunks

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import faiss
import difflib

# ══════════════════════════════════════════════════════════════
# KONFIGURASI
# ══════════════════════════════════════════════════════════════

KNOWLEDGE_BASE_DIR  = "./kb/kb_new"
VECTOR_STORE_PATH   = "./vector_store/faiss_kb"
ICD10_VECTOR_STORE  = "./vector_store/faiss_icd10"
EMBEDDING_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_ID        = "qwen2.5:3b-instruct"
ICD10_TXT_PATH      = "./kb/icd10_structured.txt"

# ── Chunking KB ────────────────────────────────────────────────
CHUNK_SIZE          = 256
CHUNK_OVERLAP       = 0

# ── Chunking ICD-10 ────────────────────────────────────────────
ICD10_CHUNK_SIZE    = 100
ICD10_CHUNK_OVERLAP = 0

# ── Retrieval ──────────────────────────────────────────────────
TOP_K_DOCS          = 10
TOP_K_ICD10         = 1
ICD10_SIMILARITY_THRESHOLD = 0.8


# ══════════════════════════════════════════════════════════════
# PROMPT TEMPLATE
# ══════════════════════════════════════════════════════════════

# PERUBAHAN: instruksi "ONLY on context" → "prioritize context"
# Alasan: model 3B yang dipaksa HANYA menggunakan context fragmentaris
# justru menghasilkan reasoning yang salah (menyalin chunk tidak relevan).
# Instruksi baru meminta model memprioritaskan context tapi boleh
# melengkapi dengan pengetahuan klinis umum jika context tidak cukup.
# Ini lebih jujur secara akademis dan menghasilkan output lebih baik.

DIAGNOSIS_PROMPT = DIAGNOSIS_PROMPT = """You are a sports medicine physician assistant specializing in running injuries. Below are examples of how to analyze a clinical presentation and generate 3 differential diagnoses with supporting clinical information.

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

**Distinguishing Features:**
The key differentiator between these three diagnoses is the location and character of tenderness on physical examination. ITBS produces tenderness specifically at the lateral femoral epicondyle with pain triggered at a consistent running distance, whereas lateral meniscus tear would typically produce joint line tenderness with mechanical symptoms (locking, catching). Biceps femoris tendinopathy localises to the fibular head rather than the epicondyle. The absence of swelling and joint locking in this case argues strongly against intra-articular pathology.

**Recommended Confirmatory Tests:**
- Ober Test: assess IT band tightness — positive result (limited hip adduction) supports ITBS
- Noble Compression Test: apply pressure at lateral femoral epicondyle with knee at 30° flexion — reproduction of pain confirms ITBS
- McMurray Test: if ITBS is not confirmed, perform to evaluate for meniscal pathology
- Thessaly Test: weight-bearing meniscal provocation test — positive result shifts diagnosis toward lateral meniscus tear


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

**Distinguishing Features:**
The morning first-step pain that improves with activity is pathognomonic for plantar fasciitis and is the primary differentiator from calcaneal stress fracture, which would worsen with weight-bearing activity throughout the day. Tarsal tunnel syndrome is distinguished by the presence of neurological symptoms (numbness, tingling, burning) in the plantar foot distribution — features entirely absent in this case. Palpation is key: plantar fasciitis produces point tenderness at the medial calcaneal tubercle, whereas calcaneal stress fracture produces positive squeeze test (lateral compression of the calcaneus eliciting pain).

**Recommended Confirmatory Tests:**
- Windlass Test: passive dorsiflexion of great toe — reproduction of plantar heel pain confirms plantar fasciitis
- Calcaneal Squeeze Test: lateral compression of calcaneus — positive result raises suspicion for calcaneal stress fracture and warrants imaging
- Tinel's Sign at Tarsal Tunnel: percussion over the tibial nerve posterior to medial malleolus — tingling or electric sensation indicates tarsal tunnel syndrome
- If stress fracture suspected: X-ray (may be negative early); MRI is gold standard for early detection


IMPORTANT INSTRUCTIONS:
- Always provide EXACTLY 3 differential diagnoses, numbered 1 to 3.
- Always use the format: bold diagnosis name, then Supporting findings, Against, and Clinical reasoning.
- After the 3 diagnoses, always include a "Distinguishing Features" section and a "Recommended Confirmatory Tests" section.
- Supporting findings: cite 2-3 specific clinical findings from the presentation that support this diagnosis.
- Against: cite 1-2 specific absent findings or clinical features that argue against this diagnosis.
- Clinical reasoning: one concise sentence summarising the diagnostic conclusion.
- Recommended Confirmatory Tests: list 3-4 specific physical examination tests the clinician can perform immediately, with brief description of positive result interpretation.
- Prioritize the medical context provided below as your primary reference.
- If the context does not fully address the presentation, supplement with established sports medicine knowledge.
- Do NOT include treatment plans or management. Only differential diagnosis and clinical decision support information.
- Do NOT begin your response with preamble phrases such as "Based on the provided clinical presentation". Start directly with "1."


Medical context from knowledge base:
{context}

Clinical presentation:
{question}

Top 3 Differential Diagnosis:
"""


# ══════════════════════════════════════════════════════════════
# RAG PIPELINE
# ══════════════════════════════════════════════════════════════

class RAGDiagnosisPipeline:

    def __init__(self):
        self.embeddings   = None
        self.kb_store     = None
        self.icd10_store  = None
        self.llm          = None
        self.prompt       = None
        self._initialized = False

    # ── Embeddings ─────────────────────────────────────────────
    def _load_embeddings(self) -> HuggingFaceEmbeddings:
        print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL}")
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # ── FAISS helper ───────────────────────────────────────────
    def _make_faiss(self, chunks: List[Document], label: str) -> FAISS:
        """Buat FAISS IndexFlatIP (cosine similarity) dari chunks."""
        dim   = len(self.embeddings.embed_query("test"))
        index = faiss.IndexFlatIP(dim)
        vs = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            normalize_L2=True,
        )
        vs.add_documents(chunks)
        print(f"[INFO] {label}: FAISS ready — dim={dim}, chunks={len(chunks)}")
        return vs

    # ── Knowledge Base store ───────────────────────────────────
    def build_kb_store(self, force_rebuild: bool = False) -> FAISS:
        idx_path = Path(VECTOR_STORE_PATH)
        if idx_path.exists() and not force_rebuild:
            print("[INFO] Loading existing KB vector store...")
            return FAISS.load_local(VECTOR_STORE_PATH, self.embeddings,
                                    allow_dangerous_deserialization=True)

        print("[INFO] Building KB vector store...")
        kb_path = Path(KNOWLEDGE_BASE_DIR)
        if not kb_path.exists():
            raise FileNotFoundError(f"KB not found: {KNOWLEDGE_BASE_DIR}")

        raw_docs: List[Document] = []

        if kb_path.is_file():
            if kb_path.suffix.lower() == ".pdf":
                raw_docs = PyMuPDFLoader(str(kb_path)).load()
                print(f"[INFO] PDF dimuat: {kb_path.name}")
            else:
                raw_docs = TextLoader(str(kb_path), encoding="utf-8").load()
                print(f"[INFO] TXT dimuat: {kb_path.name}")

        elif kb_path.is_dir():
            txt_docs: List[Document] = []
            pdf_docs: List[Document] = []

            for txt_file in kb_path.rglob("*.txt"):
                try:
                    docs = TextLoader(str(txt_file), encoding="utf-8").load()
                    txt_docs.extend(docs)
                    print(f"[INFO] TXT dimuat: {txt_file.name} ({len(docs)} doc)")
                except Exception as e:
                    print(f"[WARN] Gagal baca {txt_file.name}: {e}")

            for pdf_file in kb_path.rglob("*.pdf"):
                try:
                    docs = PyMuPDFLoader(str(pdf_file)).load()
                    pdf_docs.extend(docs)
                    print(f"[INFO] PDF dimuat: {pdf_file.name} ({len(docs)} halaman)")
                except Exception as e:
                    print(f"[WARN] Gagal baca {pdf_file.name}: {e}")

            raw_docs = txt_docs + pdf_docs

        if not raw_docs:
            raise ValueError(
                f"No documents found at: {KNOWLEDGE_BASE_DIR}\n"
                "Pastikan folder kb/ berisi file .txt atau .pdf"
            )
        print(f"[INFO] Total {len(raw_docs)} document(s) dimuat.")

        # Step 1: Preprocessing — bersihkan noise struktural PDF
        raw_docs = preprocess_documents(raw_docs)

        # Step 2: Chunking
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "],
        ).split_documents(raw_docs)

        # Step 3: Filter chunk noise
        chunks = filter_chunks(chunks)

        vs = self._make_faiss(chunks, "KB")
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        vs.save_local(VECTOR_STORE_PATH)
        print(f"[INFO] KB store saved → {VECTOR_STORE_PATH}")
        return vs

    # ── ICD-10 store ───────────────────────────────────────────
    def build_icd10_store(self, force_rebuild: bool = False) -> FAISS:
        idx_path = Path(ICD10_VECTOR_STORE)
        if idx_path.exists() and not force_rebuild:
            print("[INFO] Loading existing ICD-10 vector store...")
            return FAISS.load_local(ICD10_VECTOR_STORE, self.embeddings,
                                    allow_dangerous_deserialization=True)

        print("[INFO] Building ICD-10 vector store from structured TXT...")
        txt_path = Path(ICD10_TXT_PATH)
        if not txt_path.exists():
            raise FileNotFoundError(f"ICD-10 TXT not found: {ICD10_TXT_PATH}")

        raw = TextLoader(str(txt_path), encoding="utf-8").load()[0].page_content
        lines = [
            line.strip() for line in raw.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        line_docs = [Document(page_content=line) for line in lines]
        print(f"[INFO] ICD-10: {len(line_docs)} entri dimuat.")

        vs = self._make_faiss(line_docs, "ICD-10")
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        vs.save_local(ICD10_VECTOR_STORE)
        print(f"[INFO] ICD-10 store saved → {ICD10_VECTOR_STORE}")
        return vs

    # ── LLM ────────────────────────────────────────────────────
    def _load_llm(self) -> OllamaLLM:
        print(f"[INFO] Connecting to Ollama LLM: {LLM_MODEL_ID}")
        return OllamaLLM(
            model=LLM_MODEL_ID,
            temperature=0.3,   # PERUBAHAN: 0.3→0.1 untuk output lebih konsisten
        )

    # ── Initialize ─────────────────────────────────────────────
    def initialize(self, force_rebuild_index: bool = False):
        self.embeddings  = self._load_embeddings()
        self.kb_store    = self.build_kb_store(force_rebuild=force_rebuild_index)
        self.icd10_store = self.build_icd10_store(force_rebuild=force_rebuild_index)
        self.llm         = self._load_llm()
        self.prompt      = PromptTemplate(
            template=DIAGNOSIS_PROMPT, input_variables=["context", "question"]
        )
        self._initialized = True
        print("[INFO] RAG Pipeline ready.")

    # ── ICD-10 Validator (Fuzzy Text Matching) ──────────────────
    def _lookup_icd10(self, diagnosis_name: str) -> List[Dict[str, str]]:
        """
        Mencari kode ICD-10 menggunakan pencocokan teks ejaan (Fuzzy Matching).
        """
        results  = []
        txt_path = Path(ICD10_TXT_PATH)
        if not txt_path.exists():
            print(f"[ERROR] File {ICD10_TXT_PATH} tidak ditemukan.")
            return results

        raw_lines = txt_path.read_text(encoding="utf-8").split("\n")

        db_diagnoses = {}
        for line in raw_lines:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 2:
                    db_diagnoses[parts[0]] = parts[1]

        matches = difflib.get_close_matches(
            word=diagnosis_name,
            possibilities=db_diagnoses.keys(),
            n=TOP_K_ICD10,
            cutoff=0.3,
        )

        seen = set()
        for match in matches:
            code = db_diagnoses[match]
            if code not in seen:
                results.append({"code": code, "description": match})
                seen.add(code)

        return results

    # ── Ekstrak nama diagnosis dari output LLM ──────────────────
    def _extract_diagnosis_names(self, llm_text: str) -> List[str]:
        """
        Ekstrak nama-nama diagnosis dari output teks LLM.
        """
        names = []

        marker      = re.search(r'Differential Diagnosis\s*:\s*', llm_text, re.IGNORECASE)
        search_text = llm_text[marker.end():] if marker else llm_text

        # Pola 1: "1. Name" atau "1) Name"
        # Tambah apostrof (') agar nama seperti "Jumper's Knee" ter-ekstrak
        for m in re.finditer(
            r"^[ \t]*\d+[\.)][\.)\s]{0,3}\*{0,2}([A-Z][A-Za-z\(\)\-\/][A-Za-z\(\)\-\/.' ]{2,58}?)\*{0,2}[ \t]*$",
            search_text, re.MULTILINE
        ):
            raw  = m.group(1).strip().rstrip(':').strip()
            name = re.split(r'  +|\t', raw)[0].strip()
            if name and len(name) > 4 and name not in names:
                names.append(name)

        # Pola 2: **Bold Name**
        # Tambah apostrof (') agar nama seperti "Jumper's Knee" ter-ekstrak
        for m in re.finditer(r"\*\*([A-Z][A-Za-z\(\)\-\/.' ]{5,60}?)\*\*", search_text):
            name = m.group(1).strip()
            if name and name not in names:
                names.append(name)

        return names[:4]

    # ── Query ──────────────────────────────────────────────────
    def query(self, question: str) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized")

        # Retrieve KB context
        kb_docs    = self.kb_store.similarity_search(question, k=TOP_K_DOCS)
        kb_context = "\n\n".join(d.page_content for d in kb_docs)

        # LLM generate diagnosis
        final_prompt   = self.prompt.format(context=kb_context, question=question)
        diagnosis_text = self.llm.invoke(final_prompt)

        # ICD-10 Validator
        diag_names = self._extract_diagnosis_names(diagnosis_text)
        print(f"[INFO] Extracted diagnoses: {diag_names}")

        icd10_validated: Dict[str, List[Dict]] = {}
        for name in diag_names:
            codes = self._lookup_icd10(name)
            icd10_validated[name] = codes
            print(f"[INFO] ICD-10 for '{name}': {[c['code'] for c in codes]}")

        return {
            "diagnosis_text"  : diagnosis_text,
            "icd10_validated" : icd10_validated,
            "kb_docs"         : [
                {"content": d.page_content, "source": d.metadata.get("source", "-")}
                for d in kb_docs
            ],
        }

    # ── Tambah dokumen baru ─────────────────────────────────────
    def add_documents(self, file_paths: List[str]):
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized.")
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "],
        ).split_documents([
            doc for fp in file_paths
            for doc in TextLoader(fp, encoding="utf-8").load()
        ])
        self.kb_store.add_documents(chunks)
        self.kb_store.save_local(VECTOR_STORE_PATH)
        print(f"[INFO] {len(chunks)} new chunks added.")


# ══════════════════════════════════════════════════════════════
# BLOK TESTING
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import textwrap

    TEST_QUERIES = [
        (
            "Query",
            "A 24-year-old male member of a university running club presents with pain along his right shin that has been present for approximately one month. Initially, the pain occurred only after high-intensity interval running sessions, but during the past two weeks it has also been noticeable during long walks. The patient describes the pain as a diffuse aching sensation extending along the inner border of the shin and occasionally accompanied by a burning sensation. The symptoms began shortly after he significantly increased the intensity and frequency of his training in preparation for a university competition. The patient denies any direct trauma to the lower leg. Physical examination reveals tenderness along the distal medial border of the tibia without obvious swelling or deformity. The pain intensifies during light jumping activities."
        )  
    ]

    # !! PENTING: set True setiap kali kamu mengubah KB atau parameter chunking
    # Set False setelah rebuild pertama agar tidak rebuild ulang setiap run
    FORCE_REBUILD_INDEX = False
    SHOW_KB_DOCS        = True
    PREVIEW_LEN         = 300   # dinaikkan agar chunk yang lebih besar terbaca

    pipeline = RAGDiagnosisPipeline()
    pipeline.initialize(force_rebuild_index=FORCE_REBUILD_INDEX)

    for idx, (label, query_text) in enumerate(TEST_QUERIES, start=1):
        print(f"\n{'─' * 65}")
        print(f"\n COMPLAINT:\n{textwrap.fill(query_text, width=65)}\n")

        try:
            result = pipeline.query(query_text)

            print(" DIFFERENTIAL DIAGNOSIS (LLM):")
            print("-" * 65)
            print(result["diagnosis_text"].strip())

            if result["icd10_validated"]:
                print(f"\n  ICD-10 VALIDATED CODES:")
                print("-" * 65)
                for diag_name, codes in result["icd10_validated"].items():
                    print(f"\n  Diagnosis : {diag_name}")
                    if codes:
                        for c in codes:
                            print(f"  ICD-10    : {c['code']:12s} — {c['description']}")
                    else:
                        print(f"  ICD-10    : Not found in database")

            if SHOW_KB_DOCS and result["kb_docs"]:
                print(f"\n KB REFERENCES ({len(result['kb_docs'])} chunks):")
                for i, doc in enumerate(result["kb_docs"], 1):
                    fname   = doc["source"].split("/")[-1].split("\\")[-1]
                    preview = doc["content"][:PREVIEW_LEN].replace("\n", " ")
                    print(f"  [{i}] {fname}")
                    print(f"      {preview}{'...' if len(doc['content']) > PREVIEW_LEN else ''}")

        except Exception as e:
            print(f" ERROR on query {idx}: {e}")

    print(f"\n{'=' * 65}")
    print("  Testing complete.")
    print("=" * 65)