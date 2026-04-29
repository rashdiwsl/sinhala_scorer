# 🏛️ Offline Intelligent Sinhala Open-Ended Answer Scorer

**Natural Language Processing — Individual Assignment 2**  
General Sir John Kotelawala Defence University · Faculty of Computing · Department of Computer Science

> An intelligent, fully offline system that scores Sinhala open-ended answers for History of Sri Lanka (Anuradhapura Period) questions — graded out of 20 marks with explainable, evidence-grounded feedback.

---

## 📌 Project Overview

This system evaluates student answers written in Sinhala against a structured marking guide, using a four-agent NLP pipeline powered by a local LLM. All inference, retrieval, and processing runs entirely offline — no internet connection is required at runtime.

**Topic Scope:** Ancient Sri Lanka — Anuradhapura Period (~380 BCE – 1017 CE)

The system covers four thematic areas:
- **Administration** — central monarchy, Ministerial Councils, provincial governance
- **Irrigation & Civilisation** — tank irrigation systems
- **Buddhism & Culture** — introduction of Buddhism, stupa construction, literary works
- **Notable Rulers & Events** — Devanampiya Tissa, Dutugamunu, Valagamba, Mahasena

---

## 🎬 Demo

📺 **Demo Video:** [Watch on Google Drive](https://drive.google.com/file/d/1PYI-WlwHCqSlDNhc8bdN3Pc7FxCFNkvH/view?usp=drive_link)  
*Recorded with Wi-Fi disabled to verify full offline operation. Best viewed in 1080p.*

---

## 🗂️ Project Structure

```
├── app.py                          # Main Streamlit application
├── knowledge_base/
│   └── anuradhapura.txt            # 18 Sinhala-language knowledge sentences
├── ontology/
│   └── ontology.json               # Sinhala concept ontology (5 categories, 16 terms)
├── chroma_db/                      # Persistent ChromaDB vector store (auto-generated)
└── requirements.txt
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| UI Framework | Streamlit |
| Local LLM | gemma3:1b via OLLAMA |
| Embedding Model | all-MiniLM-L6-v2 (SentenceTransformers) |
| Vector Store | ChromaDB (PersistentClient) |
| Knowledge Base | Plain text (18 Sinhala sentences) |
| Ontology | JSON (5 categories, 4 ruler-achievement relations) |

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.9+
- [OLLAMA](https://ollama.com/) installed and running locally

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd sinhala-answer-scorer
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull the LLM model via OLLAMA

```bash
ollama pull gemma3:1b
```

> The `all-MiniLM-L6-v2` SentenceTransformer model will be automatically downloaded and cached by HuggingFace on first run.

### 4. Run the application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

> **Note:** After the first run, ChromaDB embeddings are cached in `chroma_db/`. Subsequent launches are faster and require no re-embedding.

---

## 🧠 System Architecture

The system implements a sequential **four-agent pipeline**:

```
Student Input
     │
     ▼
┌─────────────────────────────────────┐
│         Layer 1 — Streamlit UI      │
│  Question selector · Answer input   │
│  Word counter · Live agent status   │
└──────────────────┬──────────────────┘
                   │
     ┌─────────────▼─────────────┐
     │   Layer 2 — Agent Pipeline │
     │                            │
     │  Agent 1: Retrieval        │  ← Queries ChromaDB (top-3 passages)
     │  Agent 2: Ontology         │  ← Extracts Sinhala concept hints
     │  Agent 3: Scoring (LLM)    │  ← Calls gemma3:1b via OLLAMA
     │  Agent 4: Explanation      │  ← Parses, clamps, structures output
     └─────────────┬──────────────┘
                   │
     ┌─────────────▼─────────────┐
     │    Layer 3 — Data Stores   │
     │  ChromaDB · ontology.json  │
     │  anuradhapura.txt          │
     └─────────────┬──────────────┘
                   │
     ┌─────────────▼─────────────┐
     │     Layer 4 — Output       │
     │  Score /20 · Breakdown     │
     │  Sinhala justification     │
     │  Evidence cards            │
     └────────────────────────────┘
```

### Agent Responsibilities

| Agent | Input | Output |
|---|---|---|
| Agent 1 — Retrieval | Question + `answer[:200]` | Top-3 KB passages from ChromaDB |
| Agent 2 — Ontology | Question key (Q1–Q5) | Category:concept hint lines |
| Agent 3 — Scoring | Question, guide, docs, answer, hints | Raw SCORES / TOTAL / REASON |
| Agent 4 — Explanation | Raw model text + criteria list | Structured dict with scores + justification |

---

## 📋 Questions & Marking Guides

Five open-ended questions, each graded **/20**, all in Sinhala:

| # | Topic | English Translation |
|---|---|---|
| Q1 | දේවානම්පියතිස්ස රජු හා බෞද්ධාගම | Introduction of Buddhism — King Devanampiya Tissa |
| Q2 | දුටුගැමුණු රජු | King Dutugamunu's contributions |
| Q3 | ජල කළමනාකරණ පද්ධතිය | Irrigation & water management system |
| Q4 | පරිපාලන ක්‍රමය | Administrative system |
| Q5 | මහාවංශය සහ දීපවංශය | Importance of Mahavamsa & Dipavamsa |

---

## 🗃️ Knowledge Base & Ontology

### Knowledge Base (`anuradhapura.txt`)
18 curated Sinhala-language sentences covering all five thematic areas. Embedded at startup using `all-MiniLM-L6-v2` and stored in ChromaDB.

### Ontology (`ontology.json`)

| Category | Concepts |
|---|---|
| රජවරු (Rulers) | දේවානම්පියතිස්ස, දුටුගැමුණු, වළගම්බා, මහාසේන |
| ආගම (Religion) | බෞද්ධාගම, මහින්ද හිමි, ත්‍රිපිටකය, තූපාරාමය |
| ජල කළමනාකරණය (Irrigation) | නුවර වැව, තිස්ස වැව, අභය වැව, පරාක්‍රම සමුද්‍රය |
| ස්තූප (Stupas) | රුවන්වැලිසාය, ජේතවනාරාමය, අභයගිරිය |
| සිදුවීම් (Events) | බෞද්ධාගම හඳුන්වාදීම, දුටුගැමුණු-එළාර සටන, මහාවංශය ලිවීම |

Relations map links each ruler to their key achievements (built / embraced / defeated / commissioned).

---

## 📊 Scoring & Explainability

Every submission produces five output components:

| Output | Description |
|---|---|
| **Final score /20** | Colour-coded: green (≥15), amber (8–14), red (<8) |
| **Per-criterion breakdown** | Awarded/max marks with colour-coded progress bars |
| **Sinhala justification** | 2-sentence explanation of criteria met/missed |
| **Evidence cards (×3)** | Top-3 retrieved KB passages with ontology terms highlighted in gold |
| **Session history** | All Q/score pairs from the current session (sidebar) |

### Score Ceiling by Answer Length

| Word Count | Max Possible Score |
|---|---|
| < 20 words | 6/20 |
| 20–49 words | 10/20 |
| 50–99 words | 14/20 |
| 100+ words | 20/20 (accuracy-dependent) |

---

## 🔌 Offline Operation

All components run without any network access:

| Component | Offline Mechanism |
|---|---|
| gemma3:1b | Weights stored locally; `ollama.chat()` calls local daemon |
| all-MiniLM-L6-v2 | Cached locally by HuggingFace after first download |
| ChromaDB | PersistentClient with local `chroma_db/` path |
| Knowledge base | Read from `knowledge_base/anuradhapura.txt` |
| Ontology | Read from `ontology/ontology.json` |

---

## 🖥️ UI Features

The Streamlit interface uses a custom **ancient-stone and gold theme** and is divided into three zones:

- **Left panel** — Question selector, Sinhala answer input, live word counter, Evaluate button
- **Right panel** — Always-visible marking guide with criteria and mark allocations
- **Results section** — Live agent pipeline status, score box, Sinhala justification, per-criterion breakdown, evidence cards
- **Sidebar** — Session history of all scored answers

---

## 👤 Author

**RMRBD Rathnayake** 