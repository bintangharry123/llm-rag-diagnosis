import streamlit as st
import time
import sys
import os

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="RunDx — Differential Diagnosis System",
    page_icon="🏃",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Hide sidebar & default streamlit chrome ────────────────────
st.markdown("""
<style>
/* Hide sidebar completely */
[data-testid="stSidebar"] { display: none; }
[data-testid="collapsedControl"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* Root variables */
:root {
    --bg:         #0f1117;
    --surface:    #161b27;
    --surface2:   #1e2535;
    --border:     #2a3448;
    --accent:     #4f8ef7;
    --accent-dim: #1d3461;
    --success:    #34d399;
    --text:       #e2e8f0;
    --text-dim:   #64748b;
    --text-muted: #94a3b8;
    --red:        #f87171;
    --font-serif: 'DM Serif Display', Georgia, serif;
    --font-mono:  'DM Mono', 'Courier New', monospace;
    --font-sans:  'DM Sans', sans-serif;
    --radius:     12px;
    --radius-lg:  20px;
}

/* Global */
.stApp {
    background: var(--bg);
    font-family: var(--font-sans);
}

.block-container {
    max-width: 780px !important;
    padding-top: 3rem !important;
    padding-bottom: 4rem !important;
}

/* ── Header ─────────────────────────────────────── */
.rundx-header {
    text-align: center;
    margin-bottom: 2.5rem;
}

.rundx-wordmark {
    font-family: var(--font-serif);
    font-size: 3rem;
    font-weight: 400;
    color: var(--text);
    letter-spacing: -0.02em;
    line-height: 1;
    margin: 0;
}

.rundx-wordmark span {
    color: var(--accent);
    font-style: italic;
}

.rundx-subtitle {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-dim);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 0.6rem;
}

.rundx-disclaimer {
    display: inline-block;
    background: #1a1f2e;
    border: 1px solid #2a3448;
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-dim);
    letter-spacing: 0.04em;
    margin-top: 1rem;
    text-align: left;
    width: 100%;
}

/* ── Input area ─────────────────────────────────── */
.input-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 500;
    color: var(--text-dim);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
    caret-color: var(--accent) !important;
    transition: border-color 0.2s ease !important;
    resize: vertical !important;
}

textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79, 142, 247, 0.12) !important;
    outline: none !important;
}

textarea::placeholder {
    color: var(--text-dim) !important;
    font-style: italic;
}

/* ── Button ─────────────────────────────────────── */
.stButton > button {
    width: 100%;
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.85rem 2rem !important;
    cursor: pointer !important;
    transition: background 0.2s ease, transform 0.1s ease !important;
    margin-top: 0.5rem;
}

.stButton > button:hover {
    background: #3b7de8 !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

.stButton > button:disabled {
    background: var(--surface2) !important;
    color: var(--text-dim) !important;
    cursor: not-allowed !important;
    transform: none !important;
}

/* ── Output container ───────────────────────────── */
.output-wrapper {
    margin-top: 2rem;
}

.output-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
}

.output-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 500;
    color: var(--text-dim);
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.output-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--success);
    animation: pulse 2s ease infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Diagnosis blocks */
.diag-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    position: relative;
}

.diag-number {
    position: absolute;
    top: -0.75rem;
    left: 1rem;
    background: var(--accent);
    color: #fff;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 500;
    padding: 0.15rem 0.55rem;
    border-radius: 100px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.diag-name {
    font-family: var(--font-serif);
    font-size: 1.2rem;
    color: var(--text);
    margin: 0.25rem 0 0.75rem 0;
}

.icd-badge {
    display: inline-block;
    background: var(--accent-dim);
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: 0.65rem;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    margin-left: 0.5rem;
    letter-spacing: 0.05em;
    vertical-align: middle;
}

.field-label {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--accent);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
    margin-top: 0.8rem;
}

.field-content {
    font-family: var(--font-sans);
    font-size: 0.875rem;
    color: var(--text-muted);
    line-height: 1.65;
    margin: 0;
}

.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 0.75rem 0;
}

/* Extra sections (Distinguishing, Confirmatory) */
.extra-block {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

.extra-title {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-dim);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

.extra-content {
    font-family: var(--font-sans);
    font-size: 0.875rem;
    color: var(--text-muted);
    line-height: 1.7;
    white-space: pre-wrap;
}

/* Raw fallback */
.raw-output {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    font-family: var(--font-mono);
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.7;
    white-space: pre-wrap;
    overflow-x: auto;
}

/* Error box */
.error-box {
    background: rgba(248, 113, 113, 0.08);
    border: 1px solid rgba(248, 113, 113, 0.3);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--red);
}

/* Status bar */
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-dim);
    letter-spacing: 0.08em;
    margin-top: 0.5rem;
}

.spinner {
    width: 12px; height: 12px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    display: inline-block;
}

@keyframes spin { to { transform: rotate(360deg); } }

/* Divider between sections */
.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Pipeline init (cached) ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rag_pipeline import RAGDiagnosisPipeline
    pipeline = RAGDiagnosisPipeline()
    pipeline.initialize(force_rebuild_index=False)
    return pipeline


# ── Parser: pecah output LLM menjadi struktur terstruktur ──────
def parse_diagnosis_output(text: str):
    """
    Kembalikan list of dict:
      {name, supporting, against, reasoning, icd10_codes}
    dan sections tambahan {distinguishing, confirmatory}
    """
    import re

    diagnoses = []

    # Pisahkan bagian Distinguishing Features & Confirmatory Tests
    extra = {}
    dist_match = re.search(
        r'\*{0,2}Distinguishing Features\*{0,2}[\s:]*(.+?)(?=\*{0,2}Recommended Confirmatory|\Z)',
        text, re.IGNORECASE | re.DOTALL
    )
    conf_match = re.search(
        r'\*{0,2}Recommended Confirmatory Tests\*{0,2}[\s:]*(.+)',
        text, re.IGNORECASE | re.DOTALL
    )
    if dist_match:
        extra['distinguishing'] = dist_match.group(1).strip()
    if conf_match:
        extra['confirmatory'] = conf_match.group(1).strip()

    # Potong teks sebelum Distinguishing Features
    main_text = text
    if dist_match:
        main_text = text[:dist_match.start()]

    # Split per diagnosis (1. / 2. / 3.)
    blocks = re.split(r'\n(?=\d+\.\s+\*{0,2}[A-Z])', main_text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Nama diagnosis
        name_match = re.match(
            r'\d+\.\s+\*{0,2}(.+?)\*{0,2}(?:\s*\n|$)', block
        )
        if not name_match:
            continue
        name = name_match.group(1).strip().rstrip(':').strip()

        # Supporting findings
        sup = re.search(
            r'[Ss]upporting\s+findings?[:\s]+(.+?)(?=Against:|Clinical reasoning:|$)',
            block, re.DOTALL
        )
        # Against
        ags = re.search(
            r'Against[:\s]+(.+?)(?=Clinical reasoning:|$)',
            block, re.DOTALL
        )
        # Clinical reasoning
        rsn = re.search(
            r'Clinical\s+reasoning[:\s]+(.+?)(?=\Z|\n\n)',
            block, re.DOTALL
        )

        diagnoses.append({
            'name':       name,
            'supporting': sup.group(1).strip() if sup else '',
            'against':    ags.group(1).strip() if ags else '',
            'reasoning':  rsn.group(1).strip() if rsn else '',
        })

    return diagnoses, extra


# ── Render diagnosis cards ─────────────────────────────────────
def render_output(result: dict):
    text   = result['diagnosis_text']
    icd_map = result.get('icd10_validated', {})

    diagnoses, extra = parse_diagnosis_output(text)

    st.markdown('<div class="output-wrapper">', unsafe_allow_html=True)
    st.markdown('''
        <div class="output-header">
            <span class="output-dot"></span>
            <span class="output-label">Differential Diagnosis</span>
        </div>
    ''', unsafe_allow_html=True)

    if not diagnoses:
        # Fallback: tampilkan raw text
        st.markdown(f'<div class="raw-output">{text}</div>', unsafe_allow_html=True)
    else:
        for i, d in enumerate(diagnoses, 1):
            # ICD badge
            icd_html = ''
            for name_key, codes in icd_map.items():
                if codes and any(
                    w.lower() in d['name'].lower() or d['name'].lower() in name_key.lower()
                    for w in name_key.split()
                ):
                    icd_html = f'<span class="icd-badge">{codes[0]["code"]}</span>'
                    break

            st.markdown(f'''
            <div class="diag-block">
                <div class="diag-number">DDx {i}</div>
                <div class="diag-name">{d["name"]}{icd_html}</div>
                <hr class="divider">
                <div class="field-label">Supporting Findings</div>
                <p class="field-content">{d["supporting"]}</p>
                <div class="field-label">Against</div>
                <p class="field-content">{d["against"]}</p>
                <div class="field-label">Clinical Reasoning</div>
                <p class="field-content">{d["reasoning"]}</p>
            </div>
            ''', unsafe_allow_html=True)

        # Distinguishing & Confirmatory
        if extra.get('distinguishing'):
            st.markdown(f'''
            <div class="extra-block">
                <div class="extra-title">Distinguishing Features</div>
                <div class="extra-content">{extra["distinguishing"]}</div>
            </div>
            ''', unsafe_allow_html=True)

        if extra.get('confirmatory'):
            st.markdown(f'''
            <div class="extra-block">
                <div class="extra-title">Recommended Confirmatory Tests</div>
                <div class="extra-content">{extra["confirmatory"]}</div>
            </div>
            ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════

# Header
st.markdown('''
<div class="rundx-header">
    <h1 class="rundx-wordmark">Run<span>DDx</span></h1>
    <p class="rundx-subtitle">Differential Diagnosis · Running Injury · Decision Support</p>
    <div class="rundx-disclaimer">
        ⚠ &nbsp;Sistem ini merupakan alat bantu <em>screening</em> awal berbasis AI, bukan pengganti penilaian klinis tenaga medis.
        Keputusan diagnosis akhir sepenuhnya berada pada wewenang klinisi yang bersangkutan.
    </div>
</div>
''', unsafe_allow_html=True)

# Load pipeline dengan spinner
with st.spinner("Memuat sistem RAG..."):
    try:
        pipeline = load_pipeline()
        pipeline_ready = True
    except Exception as e:
        pipeline_ready = False
        st.markdown(f'<div class="error-box">⚠ Gagal memuat pipeline: {e}</div>', unsafe_allow_html=True)

# Input section
st.markdown('<div class="input-label">Clinical Vignette</div>', unsafe_allow_html=True)

vignette = st.text_area(
    label="clinical_vignette",
    label_visibility="collapsed",
    placeholder=(
        "Deskripsikan presentasi klinis pasien di sini...\n\n"
        "Contoh: Male, 28 years old, competitive runner (60 km/week). "
        "Reports sharp lateral knee pain appearing consistently after 6 km of running, "
        "aggravated by downhill running, relieved by rest. Tenderness at lateral femoral epicondyle. "
        "No swelling, no locking, no acute trauma history."
    ),
    height=200,
)

analyze_btn = st.button(
    "Analyze →",
    disabled=(not pipeline_ready or not vignette.strip()),
    use_container_width=True,
)

# Processing
if analyze_btn and vignette.strip() and pipeline_ready:
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    with st.spinner("Menganalisis vignette..."):
        t0 = time.time()
        try:
            result  = pipeline.query(vignette.strip())
            elapsed = time.time() - t0
            render_output(result)
            st.markdown(
                f'<div class="status-bar">✓ &nbsp;Selesai dalam {elapsed:.1f}s</div>',
                unsafe_allow_html=True
            )
        except Exception as e:
            st.markdown(
                f'<div class="error-box">⚠ Error saat memproses query: {e}</div>',
                unsafe_allow_html=True
            )
