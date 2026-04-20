# RAG System — AI Policy Q&A

Built a question-answering system over three AI policy documents using retrieval-augmented generation: the EU AI Act, the OECD AI Principles, and the UNESCO AI Ethics Recommendation.

---

## What it does

You ask a question in natural language. The system finds the most relevant passages from the documents, feeds them into a prompt, and gets a Hugging Face LLM to generate an answer. There's also a Gradio chat interface so you can try it interactively.

---

## Setup

```bash
pip install transformers chromadb sentence-transformers huggingface-hub \
    langchain_community langchain-text-splitters pypdf gradio tqdm accelerate
```

Put your PDF files in a `data/` folder, then run the notebook top to bottom.

---

## Structure

```
RAG_Project.ipynb   # main notebook
data/
  eu_ai_act.pdf
  oecd_ai_principles.pdf
  unesco_ai_ethics.pdf
rag_db/             # ChromaDB persistent storage (auto-created)
evaluation_charts.png
```

---

## What was compared

**Embedding models**
- `all-MiniLM-L6-v2` — fast, 384-dim
- `all-mpnet-base-v2` — slower, 768-dim, better retrieval quality

**LLMs**
- `google/flan-t5-base` — ~250 MB, answers in a few seconds on CPU
- `google/flan-t5-large` — ~800 MB, noticeably better answers, but very slow without a GPU

**Prompt templates**
- Basic, XML Tags, Expert — tested across both models

**Retrieval depth (k)**
- Tested k=1, 3, 5 — k=3 turned out optimal

---

## Results

MPNet retrieves better chunks than MiniLM, especially for technical policy questions. flan-t5-large gives much more complete answers than flan-t5-base but takes several minutes per query on CPU — fine for a demo, not for production.

The Expert prompt worked best overall. k=3 matched k=5 in answer quality but was about 27% faster to generate.

Best config: **MPNet + flan-t5-large + Expert prompt + k=3**

---

## Running the Gradio interface

Run the last cell in the notebook. A local web interface opens where you can pick the embedding model, LLM, and prompt style, ask questions, and see the retrieved context that was used to generate each answer.
