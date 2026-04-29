import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import ollama
import json
import time

# ── CONFIG ──────────────────────────────────────────────
OLLAMA_MODEL = "gemma3:1b"
KB_PATH = "knowledge_base/anuradhapura.txt"
ONTOLOGY_PATH = "ontology/ontology.json"
CHROMA_PATH = "chroma_db"

# ── PAGE CONFIG (must be first Streamlit call) ───────────
st.set_page_config(
    page_title="අනුරාධාපුර පිළිතුරු ඇගයුම",
    page_icon="🏛️",
    layout="wide"
)

# ── ANURADHAPURA THEME CSS ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=IM+Fell+English:ital@0;1&display=swap');

:root {
    --gold: #C9A84C;
    --gold-light: #E8C97A;
    --gold-dark: #8B6914;
    --stone: #1A1208;
    --stone-mid: #2D2010;
    --stone-light: #3D2E16;
    --stone-surface: #4A3820;
    --cream: #F5E6C8;
    --cream-dim: #C8B090;
    --red-accent: #8B2020;
    --lotus: #C4526B;
    /* Score colours */
    --score-green: #4CAF72;
    --score-green-bg: rgba(76,175,114,0.12);
    --score-green-border: rgba(76,175,114,0.6);
    --score-amber: #E8A020;
    --score-amber-bg: rgba(232,160,32,0.12);
    --score-amber-border: rgba(232,160,32,0.6);
    --score-red: #D04040;
    --score-red-bg: rgba(208,64,64,0.12);
    --score-red-border: rgba(208,64,64,0.6);
}

.stApp {
    background-color: var(--stone);
    background-image:
        radial-gradient(ellipse at 20% 20%, rgba(201,168,76,0.06) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 80%, rgba(201,168,76,0.04) 0%, transparent 50%),
        repeating-linear-gradient(0deg, transparent, transparent 40px, rgba(201,168,76,0.015) 40px, rgba(201,168,76,0.015) 41px),
        repeating-linear-gradient(90deg, transparent, transparent 40px, rgba(201,168,76,0.015) 40px, rgba(201,168,76,0.015) 41px);
    font-family: 'IM Fell English', serif;
    color: var(--cream);
}

#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 1rem !important; max-width: 1200px; }

/* ── HEADER ── */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid var(--gold-dark);
    margin-bottom: 2rem;
    position: relative;
}
.hero-header::before {
    content: "❖ ❖ ❖";
    display: block;
    color: var(--gold-dark);
    font-size: 0.9rem;
    letter-spacing: 8px;
    margin-bottom: 0.8rem;
}
.hero-title {
    font-family: 'Cinzel', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--gold);
    text-shadow: 0 0 40px rgba(201,168,76,0.3), 0 2px 4px rgba(0,0,0,0.8);
    letter-spacing: 3px;
    margin: 0;
    line-height: 1.2;
}
.hero-sub {
    font-family: 'Cinzel', serif;
    font-size: 0.95rem;
    color: var(--cream-dim);
    letter-spacing: 4px;
    margin-top: 0.5rem;
    font-weight: 400;
}
.hero-ornament {
    color: var(--gold-dark);
    font-size: 1.2rem;
    margin-top: 1rem;
    letter-spacing: 12px;
}

/* ── PANELS ── */
.stone-panel {
    background: linear-gradient(135deg, var(--stone-mid) 0%, var(--stone-light) 100%);
    border: 1px solid var(--gold-dark);
    border-radius: 2px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    position: relative;
    box-shadow: inset 0 1px 0 rgba(201,168,76,0.1), 0 4px 20px rgba(0,0,0,0.5);
}

.panel-title {
    font-family: 'Cinzel', serif;
    font-size: 0.75rem;
    letter-spacing: 4px;
    color: var(--gold);
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(201,168,76,0.2);
}

/* ── QUESTION DISPLAY ── */
.question-text {
    font-family: 'IM Fell English', serif;
    font-size: 1.25rem;
    color: var(--cream);
    line-height: 1.7;
    padding: 1rem 1.2rem;
    background: rgba(0,0,0,0.3);
    border-left: 3px solid var(--gold);
    margin-bottom: 0.8rem;
    font-style: italic;
}

/* ── MARKING GUIDE ── */
.guide-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid rgba(201,168,76,0.08);
    font-size: 0.9rem;
}
.guide-row:last-child { border-bottom: none; }
.guide-criterion { color: var(--cream-dim); flex: 1; }
.guide-marks {
    color: var(--gold);
    font-family: 'Cinzel', serif;
    font-weight: 600;
    font-size: 0.85rem;
    background: rgba(201,168,76,0.1);
    padding: 0.1rem 0.5rem;
    border: 1px solid rgba(201,168,76,0.2);
    border-radius: 2px;
    min-width: 60px;
    text-align: center;
}

/* ── SELECTBOX ── */
.stSelectbox > div > div {
    background: var(--stone-light) !important;
    border: 1px solid var(--gold-dark) !important;
    color: var(--cream) !important;
    border-radius: 2px !important;
    font-family: 'Cinzel', serif !important;
    font-size: 0.85rem !important;
}
.stSelectbox label {
    font-family: 'Cinzel', serif !important;
    color: var(--gold) !important;
    font-size: 0.75rem !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
}

/* ── TEXTAREA ── */
.stTextArea textarea {
    background: var(--stone-mid) !important;
    border: 1px solid var(--gold-dark) !important;
    color: var(--cream) !important;
    border-radius: 2px !important;
    font-family: 'IM Fell English', serif !important;
    font-size: 1rem !important;
    line-height: 1.8 !important;
    caret-color: var(--gold) !important;
}
.stTextArea textarea:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 15px rgba(201,168,76,0.15) !important;
}
.stTextArea label {
    font-family: 'Cinzel', serif !important;
    color: var(--gold) !important;
    font-size: 0.75rem !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
}

/* ── WORD COUNT ── */
.word-count-bar {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-top: 0.4rem;
    padding: 0.3rem 0.6rem;
    background: rgba(0,0,0,0.2);
    border: 1px solid rgba(201,168,76,0.1);
    border-radius: 2px;
}
.word-count-num {
    font-family: 'Cinzel', serif;
    font-size: 0.8rem;
    color: var(--gold);
    min-width: 90px;
}
.word-count-hint {
    font-size: 0.75rem;
    color: var(--cream-dim);
    font-style: italic;
}
.wc-low { color: var(--score-red) !important; }
.wc-mid { color: var(--score-amber) !important; }
.wc-good { color: var(--score-green) !important; }

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, var(--gold-dark) 0%, var(--gold) 50%, var(--gold-dark) 100%) !important;
    color: var(--stone) !important;
    font-family: 'Cinzel', serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(201,168,76,0.3) !important;
}
.stButton > button:hover {
    box-shadow: 0 4px 25px rgba(201,168,76,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── SCORE COLOUR VARIANTS ── */
.score-total {
    text-align: center;
    padding: 1.5rem;
    background: linear-gradient(135deg, rgba(201,168,76,0.15), rgba(201,168,76,0.05));
    border: 1px solid var(--gold);
    margin-bottom: 1.2rem;
}
.score-total.green {
    background: var(--score-green-bg);
    border-color: var(--score-green-border);
}
.score-total.amber {
    background: var(--score-amber-bg);
    border-color: var(--score-amber-border);
}
.score-total.red {
    background: var(--score-red-bg);
    border-color: var(--score-red-border);
}
.score-number {
    font-family: 'Cinzel', serif;
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1.1;
    white-space: nowrap;
    letter-spacing: 2px;
}
.score-number.green { color: var(--score-green); text-shadow: 0 0 30px rgba(76,175,114,0.5); }
.score-number.amber { color: var(--score-amber); text-shadow: 0 0 30px rgba(232,160,32,0.5); }
.score-number.red   { color: var(--score-red);   text-shadow: 0 0 30px rgba(208,64,64,0.5); }
.score-label {
    font-family: 'Cinzel', serif;
    font-size: 0.7rem;
    letter-spacing: 5px;
    color: var(--cream-dim);
    margin-top: 0.3rem;
}
.score-badge {
    display: inline-block;
    margin-top: 0.6rem;
    font-family: 'Cinzel', serif;
    font-size: 0.65rem;
    letter-spacing: 3px;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
}
.score-badge.green { color: var(--score-green); border: 1px solid var(--score-green-border); background: var(--score-green-bg); }
.score-badge.amber { color: var(--score-amber); border: 1px solid var(--score-amber-border); background: var(--score-amber-bg); }
.score-badge.red   { color: var(--score-red);   border: 1px solid var(--score-red-border);   background: var(--score-red-bg); }

/* ── AGENT STATUS ── */
.agent-pipeline {
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(201,168,76,0.15);
    border-radius: 2px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.agent-row {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.45rem 0;
    font-family: 'Cinzel', serif;
    font-size: 0.78rem;
    letter-spacing: 1px;
    border-bottom: 1px solid rgba(201,168,76,0.06);
    transition: all 0.3s ease;
}
.agent-row:last-child { border-bottom: none; }
.agent-icon { font-size: 0.9rem; width: 1.2rem; text-align: center; }
.agent-name { flex: 1; }
.agent-status-wait  { color: var(--cream-dim); opacity: 0.5; }
.agent-status-run   { color: var(--gold); animation: pulse 1s infinite; }
.agent-status-done  { color: var(--score-green); }
.agent-status-label {
    font-size: 0.65rem;
    letter-spacing: 2px;
    padding: 0.1rem 0.4rem;
    border-radius: 2px;
}
.lbl-wait  { color: var(--cream-dim); border: 1px solid rgba(200,176,144,0.2); opacity: 0.5; }
.lbl-run   { color: var(--gold); border: 1px solid rgba(201,168,76,0.4); background: rgba(201,168,76,0.08); }
.lbl-done  { color: var(--score-green); border: 1px solid var(--score-green-border); background: var(--score-green-bg); }

@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

/* ── BREAKDOWN ── */
.breakdown-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 0.7rem 0.8rem;
    border-bottom: 1px solid rgba(201,168,76,0.08);
    gap: 1rem;
}
.breakdown-row:last-child { border-bottom: none; }
.breakdown-criterion { color: var(--cream-dim); font-size: 0.88rem; flex: 1; }
.breakdown-score { color: var(--gold); font-family: 'Cinzel', serif; font-weight: 600; font-size: 0.85rem; white-space: nowrap; }

/* ── EVIDENCE ── */
.evidence-card {
    background: rgba(0,0,0,0.3);
    border-left: 2px solid var(--gold-dark);
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.6rem;
    font-size: 0.82rem;
    color: var(--cream-dim);
    line-height: 1.5;
}
.evidence-num {
    font-family: 'Cinzel', serif;
    color: var(--gold);
    font-size: 0.7rem;
    letter-spacing: 2px;
    margin-bottom: 0.3rem;
}
/* ontology highlight inside evidence */
.ont-highlight {
    color: var(--gold-light);
    font-weight: bold;
    background: rgba(201,168,76,0.08);
    border-radius: 2px;
    padding: 0 2px;
}

/* ── JUSTIFICATION ── */
.justification-box {
    background: rgba(201,168,76,0.05);
    border: 1px solid rgba(201,168,76,0.2);
    padding: 1rem 1.2rem;
    font-style: italic;
    color: var(--cream);
    font-size: 0.92rem;
    line-height: 1.7;
    margin-top: 0.5rem;
}

/* ── HISTORY TABLE ── */
.history-table { width: 100%; border-collapse: collapse; }
.history-table th {
    font-family: 'Cinzel', serif;
    font-size: 0.65rem;
    letter-spacing: 3px;
    color: var(--gold);
    border-bottom: 1px solid rgba(201,168,76,0.3);
    padding: 0.4rem 0.6rem;
    text-align: left;
}
.history-table td {
    font-size: 0.82rem;
    padding: 0.45rem 0.6rem;
    border-bottom: 1px solid rgba(201,168,76,0.06);
    color: var(--cream-dim);
    vertical-align: middle;
}
.history-table tr:last-child td { border-bottom: none; }
.history-score { font-family: 'Cinzel', serif; font-weight: 600; font-size: 0.85rem; }
.hs-green { color: var(--score-green); }
.hs-amber { color: var(--score-amber); }
.hs-red   { color: var(--score-red); }

/* ── ACCORDION ── */
details > summary {
    font-family: 'Cinzel', serif;
    font-size: 0.75rem;
    letter-spacing: 3px;
    color: var(--gold);
    cursor: pointer;
    padding: 0.5rem 0.4rem;
    border-bottom: 1px solid rgba(201,168,76,0.15);
    list-style: none;
    user-select: none;
}
details > summary::before { content: "▸ "; color: var(--gold-dark); }
details[open] > summary::before { content: "▾ "; }
details > summary::-webkit-details-marker { display: none; }

/* ── SPINNER ── */
.stSpinner > div { border-top-color: var(--gold) !important; }

/* ── DIVIDER ── */
.golden-divider { text-align: center; color: var(--gold-dark); letter-spacing: 8px; font-size: 0.9rem; margin: 1.5rem 0; }

/* ── FOOTER ── */
.footer-line {
    text-align: center;
    padding: 1.2rem 1rem;
    border-top: 1px solid var(--stone-light);
    margin-top: 2rem;
}
.footer-stack {
    font-family: 'Cinzel', serif;
    font-size: 0.62rem;
    letter-spacing: 3px;
    color: var(--gold-dark);
}
.footer-stack span {
    color: var(--cream-dim);
    margin: 0 0.3rem;
    font-size: 0.55rem;
    vertical-align: middle;
    opacity: 0.5;
}
</style>
""", unsafe_allow_html=True)

# ── QUESTIONS + MARKING GUIDES ───────────────────────────
QUESTIONS = {
    "Q1": {
        "text": "දේවානම්පියතිස්ස රජු සහ ශ්‍රී ලංකාවේ බෞද්ධාගම හඳුන්වාදීම ගැන විස්තර කරන්න.",
        "english": "Describe King Devanampiya Tissa and the introduction of Buddhism to Sri Lanka.",
        "guide": {
            "දේවානම්පියතිස්ස රජු ගැන හඳුනාගැනීම": 4,
            "මහින්ද හිමි ගැන සඳහන්": 4,
            "තූපාරාමය ගැන": 4,
            "බෞද්ධ සංස්කෘතියට බලපෑම": 4,
            "කාලය/සන්දර්භය": 4
        }
    },
    "Q2": {
        "text": "දුටුගැමුණු රජු ශ්‍රී ලංකා ඉතිහාසයට දේශීය ශිෂ්ටාචාරයට කළ දායකත්වය කුමක්ද?",
        "english": "What was King Dutugamunu's contribution to Sri Lankan history and civilization?",
        "guide": {
            "දුටුගැමුණු-එළාර සටන": 5,
            "රුවන්වැලිසාය ගැන": 5,
            "රාජ්‍ය ඒකාබද්ධකරණය": 5,
            "සංස්කෘතික දායකත්වය": 5
        }
    },
    "Q3": {
        "text": "අනුරාධාපුර රාජධානියේ ජල කළමනාකරණ පද්ධතිය ගැන විස්තර කරන්න.",
        "english": "Describe the irrigation/water management system of the Anuradhapura kingdom.",
        "guide": {
            "ප්‍රධාන වැව් නාම": 5,
            "ජල පද්ධතියේ වැදගත්කම": 5,
            "කෘෂිකර්මයට සම්බන්ධය": 5,
            "රජවරුන්ගේ දායකත්වය": 5
        }
    },
    "Q4": {
        "text": "අනුරාධාපුර රාජධානියේ පරිපාලන ක්‍රමය ගැන කෙටියෙන් විස්තර කරන්න.",
        "english": "Briefly describe the administrative system of the Anuradhapura kingdom.",
        "guide": {
            "කේන්ද්‍රීය රජු": 5,
            "ඇමතිවරු/නිලධාරීන්": 5,
            "ප්‍රාදේශීය පාලනය": 5,
            "ආගම සහ රාජ්‍යය": 5
        }
    },
    "Q5": {
        "text": "මහාවංශය සහ දීපවංශය ශ්‍රී ලංකා ඉතිහාසයට ඇති වැදගත්කම කුමක්ද?",
        "english": "What is the importance of Mahavamsa and Dipavamsa to Sri Lankan history?",
        "guide": {
            "මහාවංශය හඳුනාගැනීම": 5,
            "දීපවංශය හඳුනාගැනීම": 4,
            "ඓතිහාසික වටිනාකම": 6,
            "වළගම්බා/රචනා සන්දර්භය": 5
        }
    }
}

# ── RAG SETUP ────────────────────────────────────────────
@st.cache_resource
def setup_rag():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    COLLECTION_NAME = "anuradhapura-v2"
    try:
        col = client.get_collection(COLLECTION_NAME, embedding_function=ef)
    except Exception:
        col = client.create_collection(COLLECTION_NAME, embedding_function=ef)
        with open(KB_PATH, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        col.add(
            documents=lines,
            ids=[f"doc{i}" for i in range(len(lines))]
        )
    return col

@st.cache_resource
def load_ontology():
    with open(ONTOLOGY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ── HELPERS ──────────────────────────────────────────────
def score_class(total_str):
    try:
        n = int(total_str.split("/")[0])
        if n >= 15: return "green"
        if n >= 8:  return "amber"
        return "red"
    except:
        return "amber"

def score_badge_label(cls):
    return {"green": "⬆ ඉහළ ලකුණු", "amber": "◆ මධ්‍යම ලකුණු", "red": "⬇ අඩු ලකුණු"}.get(cls, "")

def highlight_ontology_terms(text, ontology):
    """Bold Sinhala ontology terms found in text"""
    try:
        all_terms = []
        for items in ontology.get("concepts", {}).values():
            all_terms.extend(items)
        for term in sorted(all_terms, key=len, reverse=True):
            if term in text:
                text = text.replace(term, f'<span class="ont-highlight">{term}</span>')
    except:
        pass
    return text

def word_count_display(text):
    words = len(text.split()) if text.strip() else 0
    if words == 0:
        cls, hint = "wc-low", "පිළිතුරු ලියන්නෙකු ආරම්භ කරන්න"
    elif words < 30:
        cls, hint = "wc-low", "ළඟ ළඟ — තව ලියන්න"
    elif words < 80:
        cls, hint = "wc-mid", "හොඳ ආරම්භයකි, තව විස්තර ලබා දෙන්න"
    else:
        cls, hint = "wc-good", "විස්තරාත්මක පිළිතුරකි"
    return words, cls, hint

# ── AGENTS ───────────────────────────────────────────────
def retrieval_agent(col, question_text, student_answer=""):
    combined_query = question_text + " " + student_answer[:200]
    results = col.query(query_texts=[combined_query], n_results=3)
    return results["documents"][0]

def ontology_agent(ontology, question_key):
    concepts = ontology["concepts"]
    hints = []
    for category, items in concepts.items():
        hints.append(f"{category}: {', '.join(items)}")
    return "\n".join(hints[:3])

def scoring_agent(question, student_answer, retrieved_docs, ontology_hints):
    criteria_list = list(question["guide"].items())
    guide_lines = "\n".join([f"{i+1}. {k} (max marks: {v})" for i, (k,v) in enumerate(criteria_list)])
    context = " | ".join(retrieved_docs[:2])

    # Measure answer depth to give the LLM a realistic ceiling
    word_count = len(student_answer.split())
    if word_count < 20:
        ceiling_hint = "This is a VERY SHORT answer (under 20 words). Maximum possible score is 6/20."
    elif word_count < 50:
        ceiling_hint = "This is a SHORT answer (under 50 words). Maximum possible score is 10/20."
    elif word_count < 100:
        ceiling_hint = "This is a MEDIUM answer (under 100 words). Maximum possible score is 14/20."
    else:
        ceiling_hint = "This is a DETAILED answer. Score according to accuracy and completeness."

    prompt = f"""You are a strict Sri Lankan history examiner. Score the student answer HONESTLY.

QUESTION: {question['english']}

MARKING CRITERIA:
{guide_lines}

REFERENCE KNOWLEDGE:
{context[:600]}

STUDENT ANSWER:
{student_answer[:800]}

IMPORTANT SCORING RULES:
- {ceiling_hint}
- Award FULL marks ONLY if the student gave SPECIFIC, CORRECT details for that criterion
- Award HALF marks if the student mentioned it vaguely or without enough detail
- Award 0 marks if the criterion was NOT mentioned at all
- Simply repeating the question or giving one-line answers should NOT get full marks
- Each criterion must be judged INDEPENDENTLY — do not give full marks just because the answer is long
- Giving 20/20 to any answer that is not truly comprehensive and accurate is FORBIDDEN

Reply in EXACTLY this format, nothing else:
SCORES:
1. [score]/[max]
2. [score]/[max]
3. [score]/[max]
4. [score]/[max]
5. [score]/[max]
TOTAL: [sum]/20
REASON: [2 sentences in Sinhala: what criteria the student met and what was missing.]"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"], criteria_list


def explanation_agent(raw_score_text, criteria_list, student_answer=""):
    import re
    lines = raw_score_text.strip().split("\n")
    criteria_scores = []
    total = None
    justification = ""
    mode = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("SCORES:"):
            mode = "scores"; continue
        if line.upper().startswith("TOTAL:"):
            mode = "total"
            m = re.search(r'(\d+)\s*/\s*20', line)
            if m: total = f"{m.group(1)}/20"
            continue
        if line.upper().startswith("REASON:"):
            mode = "reason"
            justification = line[7:].strip()
            continue
        if mode == "scores":
            m = re.match(r'^(\d+)\.\s*(\d+)\s*/\s*(\d+)', line)
            if m:
                idx = int(m.group(1)) - 1
                got = int(m.group(2))
                mx = int(m.group(3))
                if 0 <= idx < len(criteria_list):
                    name, max_marks = criteria_list[idx]
                    got = min(got, max_marks)  # cap at criterion max
                    criteria_scores.append({"name": name, "score": got, "max": max_marks})
        elif mode == "reason":
            justification += " " + line

    # Always recompute total from parsed criteria (ignore LLM's stated total)
    if criteria_scores:
        computed = sum(s["score"] for s in criteria_scores)

        # ── ANTI-INFLATION GUARD ──────────────────────────────
        # Apply a ceiling based on answer word count
        word_count = len(student_answer.split()) if student_answer.strip() else 0
        if word_count < 20:
            ceiling = 6
        elif word_count < 50:
            ceiling = 10
        elif word_count < 100:
            ceiling = 14
        else:
            ceiling = 20  # no cap for detailed answers

        # If computed total exceeds ceiling, scale down proportionally
        if computed > ceiling:
            scale = ceiling / computed
            for s in criteria_scores:
                s["score"] = round(s["score"] * scale)
            computed = sum(s["score"] for s in criteria_scores)
            # ensure we don't exceed ceiling due to rounding
            computed = min(computed, ceiling)

        total = f"{computed}/20"
    elif total is None:
        m = re.search(r'TOTAL[:\s]+(\d+)\s*/\s*20', raw_score_text, re.IGNORECASE)
        if m: total = f"{m.group(1)}/20"

    def is_english(text):
        if not text: return True
        sinhala_chars = sum(1 for c in text if '\u0D80' <= c <= '\u0DFF')
        return sinhala_chars < 5

    if is_english(justification) and criteria_scores:
        full = [s["name"] for s in criteria_scores if s["score"] == s["max"]]
        partial = [s["name"] for s in criteria_scores if 0 < s["score"] < s["max"]]
        zero = [s["name"] for s in criteria_scores if s["score"] == 0]
        parts = []
        if full:    parts.append(f"සිසුවා {'、'.join(full)} පිළිබඳ නිවැරදිව පිළිතුරු දී ඇත.")
        if partial: parts.append(f"{'、'.join(partial)} පිළිබඳ අර්ධ තොරතුරු ලබා දී ඇත.")
        if zero:    parts.append(f"{'、'.join(zero)} පිළිබඳ කිසිදු තොරතුරක් සඳහන් කර නොමැත.")
        justification = " ".join(parts) if parts else "පිළිතුරු ඇගයීම සම්පූර්ණ විය."

    return {"criteria": criteria_scores, "total": total, "justification": justification.strip()}
# ── SESSION STATE ─────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # list of {q, score, cls}

# ── HEADER ───────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🏛️ අනුරාධපුර පිළිතුරු ඇගයුම</div>
    <div class="hero-sub">පුරාණ ශ්‍රී ලංකා ඉතිහාසය · බුද්ධිමත් පිළිතුරු ඇගයීම්</div>
    <div class="hero-ornament">⚜ ☸ ⚜</div>
</div>
""", unsafe_allow_html=True)

# ── LOAD RESOURCES ────────────────────────────────────────
col_db = setup_rag()
ontology = load_ontology()

# ── SIDEBAR: SESSION HISTORY ──────────────────────────────
with st.sidebar:
    st.markdown('<div class="panel-title" style="padding-top:0.5rem;">⟡ සැසි ඉතිහාසය</div>', unsafe_allow_html=True)
    if st.session_state.history:
        rows = ""
        for h in st.session_state.history:
            rows += f'<tr><td>{h["q"]}</td><td class="history-score hs-{h["cls"]}">{h["score"]}</td></tr>'
        st.markdown(f"""
        <table class="history-table">
          <thead><tr><th>ප්‍රශ්නය</th><th>ලකුණු</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>""", unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:var(--cream-dim);font-size:0.8rem;font-style:italic;padding:0.5rem 0;">තවම ඇගයීම් නොමැත.</div>', unsafe_allow_html=True)

# ── MAIN LAYOUT ───────────────────────────────────────────
left_col, right_col = st.columns([6, 4], gap="large")

with left_col:
    q_keys = list(QUESTIONS.keys())
    selected = st.selectbox(
        "ප්‍රශ්නය තෝරන්න",
        q_keys,
        format_func=lambda k: f"{k} — {QUESTIONS[k]['text'][:40]}..."
    )
    q = QUESTIONS[selected]

    st.markdown(f'<div class="question-text">{q["text"]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel-title" style="margin-top:0.8rem;">⟡ සිංහලෙන් පිළිතුර ලියන්න</div>', unsafe_allow_html=True)
    answer = st.text_area(
        "",
        height=180,
        placeholder="මෙහි ඔබගේ පිළිතුර ලියන්න...",
        label_visibility="collapsed",
        key="answer_input"
    )

    # Live word count
    wc, wc_cls, wc_hint = word_count_display(answer)
    st.markdown(f"""
    <div class="word-count-bar">
        <span class="word-count-num {wc_cls}">වචන: {wc}</span>
        <span class="word-count-hint">{wc_hint}</span>
    </div>
    """, unsafe_allow_html=True)

    score_btn = st.button("⚜ පිළිතුර ඇගයන්න ⚜", type="primary")

with right_col:
    # Marking guide — always visible in accordion
    st.markdown('<div class="panel-title" style="margin-top:0.3rem;">⟡ ලකුණු දීමේ මාර්ගෝපදේශය — ලකුණු 20</div>', unsafe_allow_html=True)
    guide_html = ""
    for criterion, marks in q["guide"].items():
        guide_html += f"""
        <div class="guide-row">
            <span class="guide-criterion">{criterion}</span>
            <span class="guide-marks">{marks}</span>
        </div>"""
    with st.expander("▸ නිර්ණායක සහ ලකුණු බලන්න", expanded=True):
        st.markdown(guide_html, unsafe_allow_html=True)

    if not score_btn:
        st.markdown("""
        <div style="text-align:center; padding:1.5rem 1rem 1rem; margin-top:0.5rem;">
            <div style="font-size:2rem; margin-bottom:0.5rem;">☸</div>
            <div style="font-family:'Cinzel',serif; color:var(--gold); font-size:0.8rem; letter-spacing:4px; margin-bottom:0.8rem;">ඇගයීමට රැඳී සිටී</div>
            <div style="color:var(--cream-dim); font-size:0.88rem; font-style:italic; line-height:1.9;">
                ප්‍රශ්නයක් තෝරා, පිළිතුර ලියා,<br>
                <b style="color:var(--gold)">⚜ පිළිතුර ඇගයන්න ⚜</b><br>
                යන බොත්තම ඔබන්න.
            </div>
            <div style="color:var(--gold-dark); margin-top:1rem; letter-spacing:6px; font-size:0.8rem;">❖ ❖ ❖</div>
        </div>
        """, unsafe_allow_html=True)

# ── RESULTS ──────────────────────────────────────────────
if score_btn:
    if not answer.strip():
        st.warning("⚠️ කරුණාකර පළමුව ඔබගේ පිළිතුර ලියන්න.")
    else:
        st.markdown('<div class="golden-divider">— ❖ —</div>', unsafe_allow_html=True)

        # ── AGENT PIPELINE STATUS ──
        agent_placeholder = st.empty()

        def render_pipeline(states):
            """states: list of 'wait'|'run'|'done' for agents 1-4"""
            labels = {
                "wait": ("⬤", "lbl-wait", "agent-status-wait", "රැඳී සිටී"),
                "run":  ("⟳", "lbl-run",  "agent-status-run",  "ක්‍රියාත්මකයි..."),
                "done": ("✓", "lbl-done", "agent-status-done", "සම්පූර්ණයි"),
            }
            agents = [
                ("🔍", "නියෝජිත 1: RAG සෙවීම"),
                ("🧩", "නියෝජිත 2: ඔන්ටොලොජි"),
                ("⚖️", "නියෝජිත 3: ලකුණු ගණනය"),
                ("📋", "නියෝජිත 4: ප්‍රතිඵල සකස් කිරීම"),
            ]
            html = '<div class="agent-pipeline">'
            for i, (icon, name) in enumerate(agents):
                s = states[i] if i < len(states) else "wait"
                sym, lbl_cls, name_cls, lbl_text = labels[s]
                html += f"""<div class="agent-row">
                    <span class="agent-icon {name_cls}">{sym}</span>
                    <span class="agent-name {name_cls}">{icon} {name}</span>
                    <span class="agent-status-label {lbl_cls}">{lbl_text}</span>
                </div>"""
            html += '</div>'
            return html

        agent_placeholder.markdown(render_pipeline(["run","wait","wait","wait"]), unsafe_allow_html=True)
        docs = retrieval_agent(col_db, q["text"], answer)

        agent_placeholder.markdown(render_pipeline(["done","run","wait","wait"]), unsafe_allow_html=True)
        hints = ontology_agent(ontology, selected)

        agent_placeholder.markdown(render_pipeline(["done","done","run","wait"]), unsafe_allow_html=True)
        raw, criteria_list = scoring_agent(q, answer, docs, hints)

        agent_placeholder.markdown(render_pipeline(["done","done","done","run"]), unsafe_allow_html=True)
        parsed = explanation_agent(raw, criteria_list)

        agent_placeholder.markdown(render_pipeline(["done","done","done","done"]), unsafe_allow_html=True)

        # ── RESULTS COLUMNS ──
        r1, r2, r3 = st.columns([2, 4, 3], gap="large")

        with r1:
            total_display = parsed["total"] if parsed["total"] else "—/20"
            cls = score_class(total_display)
            badge = score_badge_label(cls)
            st.markdown(f"""
            <div class="panel-title">⟡ අවසාන ලකුණු</div>
            <div class="score-total {cls}">
                <div class="score-number {cls}">{total_display}</div>
                <div class="score-label">ලබාගත් ලකුණු</div>
                <div class="score-badge {cls}">{badge}</div>
            </div>
            """, unsafe_allow_html=True)
            if parsed["justification"]:
                st.markdown(f"""
                <div class="panel-title" style="margin-top:1rem;">⟡ පරීක්ෂකගේ හේතුව</div>
                <div class="justification-box">{parsed["justification"]}</div>
                """, unsafe_allow_html=True)

            # Save to session history
            already = any(h["q"] == selected and h["score"] == total_display for h in st.session_state.history)
            if not already:
                st.session_state.history.append({"q": selected, "score": total_display, "cls": cls})

        with r2:
            st.markdown('<div class="panel-title">⟡ නිර්ණායක අනුව ලකුණු විස්තරය</div>', unsafe_allow_html=True)
            if parsed["criteria"]:
                breakdown_html = ""
                for item in parsed["criteria"]:
                    pct = (item["score"] / item["max"]) * 100 if item["max"] > 0 else 0
                    bar_color = "#4CAF72" if pct >= 75 else "#E8A020" if pct >= 40 else "#D04040"
                    breakdown_html += f"""
                    <div class="breakdown-row">
                        <div style="flex:1">
                            <div class="breakdown-criterion">{item["name"]}</div>
                            <div style="height:3px;background:rgba(255,255,255,0.05);margin-top:4px;border-radius:2px;">
                                <div style="height:100%;width:{pct}%;background:{bar_color};border-radius:2px;transition:width 0.5s ease;"></div>
                            </div>
                        </div>
                        <div class="breakdown-score">{item["score"]}/{item["max"]}</div>
                    </div>"""
                st.markdown(breakdown_html, unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color:var(--cream-dim);font-size:0.85rem;white-space:pre-wrap">{raw}</div>', unsafe_allow_html=True)

        with r3:
            st.markdown('<div class="panel-title">⟡ ලබාගත් දැනුම් මූලාශ්‍ර</div>', unsafe_allow_html=True)
            for i, doc in enumerate(docs, 1):
                highlighted = highlight_ontology_terms(doc[:220], ontology)
                st.markdown(f"""
                <div class="evidence-card">
                    <div class="evidence-num">මූලාශ්‍රය {i}</div>
                    {highlighted}
                </div>""", unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────
st.markdown("""
<div class="footer-line">
    <div class="footer-stack">
        OLLAMA <span>·</span> gemma3:1b <span>·</span> ChromaDB <span>·</span> SentenceTransformers <span>·</span> Streamlit
    </div>
</div>
""", unsafe_allow_html=True)