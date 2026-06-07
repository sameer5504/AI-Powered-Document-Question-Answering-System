import os
import re
import shutil
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ── Constants ────────────────────────────────────────────────────────────────
DOCUMENTS_FOLDER = "documents"
VECTOR_FOLDER    = "vector_store"
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"


# ── Folder setup ─────────────────────────────────────────────────────────────
def setup_folders():
    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
    os.makedirs(VECTOR_FOLDER, exist_ok=True)


# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Remove common PDF artefacts (headers, footers, repeated watermarks)
    using pattern-based rules instead of hardcoded strings.
    """
    # Strip repeated short lines that look like page headers / footers
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip very short repetitive lines (page numbers, watermarks)
        if len(stripped) < 4:
            continue
        # Skip lines that look like "Page X of Y" or just a number
        if re.fullmatch(r"(page\s*)?\d+(\s*of\s*\d+)?", stripped, re.IGNORECASE):
            continue
        cleaned.append(stripped)

    text = " ".join(cleaned)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── PDF loading ───────────────────────────────────────────────────────────────
def load_pdfs():
    documents = []
    for file in os.listdir(DOCUMENTS_FOLDER):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(DOCUMENTS_FOLDER, file)
            loader    = PyPDFLoader(file_path)
            loaded    = loader.load()
            for doc in loaded:
                doc.page_content        = clean_text(doc.page_content)
                doc.metadata["source_file"] = file
            documents.extend(loaded)
    return documents


# ── Embeddings (cached across reruns) ────────────────────────────────────────
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# ── Vector store ──────────────────────────────────────────────────────────────
def create_vector_store(chunk_size: int = 700, chunk_overlap: int = 120):
    documents = load_pdfs()
    if not documents:
        st.error("No PDF files found. Upload PDFs first.")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks     = splitter.split_documents(documents)
    embeddings = get_embeddings()

    if os.path.exists(VECTOR_FOLDER):
        shutil.rmtree(VECTOR_FOLDER)

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(VECTOR_FOLDER)
    return vs


def load_vector_store():
    embeddings  = get_embeddings()
    index_file  = os.path.join(VECTOR_FOLDER, "index.faiss")
    if os.path.exists(index_file):
        return FAISS.load_local(
            VECTOR_FOLDER,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return None


# ── LLM via Ollama ────────────────────────────────────────────────────────────
def call_ollama(context: str, question: str, history: list[dict]) -> str | None:
    """
    Call local Ollama with the conversation history + current question.
    Returns the model's reply, or None if Ollama is unavailable.
    """
    try:
        import ollama

        system_prompt = """
        You are a helpful AI study assistant.

        Answer the question using only the context below.

        Requirements:
        - Give a direct answer first.
        - Explain the concept in simple language.
        - If the context contains examples, include one.
        - If the question is technical, provide a short practical example.
        - Do not copy large chunks of text.
        - Summarize the information naturally.
        - Maximum 150 words.
        - If the context does not contain enough information, say so clearly.
        """

        messages = [{"role": "system", "content": system_prompt}]

        # Include previous turns so the model has conversation memory
        for turn in history:
            messages.append({"role": "user",      "content": turn["question"]})
            messages.append({"role": "assistant",  "content": turn["answer"]})

        # Append current question with retrieved context
        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}",
        })

        response = ollama.chat(model="llama3", messages=messages)
        return response["message"]["content"]

    except Exception as e:
        st.warning(f"Ollama unavailable ({e}). Showing raw retrieved chunks instead.")
        return None


# ── Answer builder ────────────────────────────────────────────────────────────
def format_sources(docs: list) -> str:
    lines = []
    for i, doc in enumerate(docs, 1):
        page   = doc.metadata.get("page", "?")
        source = doc.metadata.get("source_file", "unknown")
        page_n = page + 1 if isinstance(page, int) else page
        lines.append(f"- **Source {i}:** {source}, page {page_n}")
    return "\n".join(lines)


def answer_question(
    vector_store,
    question: str,
    history: list[dict],
    use_llm: bool,
    top_k: int = 4,
) -> str:
    results = vector_store.similarity_search(question, k=top_k)
    if not results:
        return "No relevant information found in the uploaded PDFs."

    context = "\n\n".join([clean_text(doc.page_content) for doc in results])

    if use_llm:
        llm_reply = call_ollama(context, question, history)
        if llm_reply:
            return llm_reply + "\n\n---\n**Sources**\n" + format_sources(results)

    # Fallback: return raw chunks with source labels
    answer = ""
    for i, doc in enumerate(results, 1):
        page   = doc.metadata.get("page", "?")
        source = doc.metadata.get("source_file", "unknown")
        page_n = page + 1 if isinstance(page, int) else page
        answer += f"**Chunk {i}** — *{source}, page {page_n}*\n\n"
        answer += clean_text(doc.page_content) + "\n\n---\n"
    return answer


# ── Evaluation ────────────────────────────────────────────────────────────────
DEFAULT_EVAL_PAIRS = [
    {
        "question": "What is referential integrity?",
        "keywords": ["foreign key", "reference", "constraint"],
    },
    {
        "question": "What is a natural join?",
        "keywords": ["common attributes", "equi-join", "schema"],
    },
    {
        "question": "What is the difference between inner join and outer join?",
        "keywords": ["null", "unmatched", "preserve"],
    },
]


def run_evaluation(vector_store, top_k: int = 4) -> dict:
    """
    For each eval pair, check whether any retrieved chunk contains
    at least one expected keyword. Returns hit-rate as a simple metric.
    """
    pairs = DEFAULT_EVAL_PAIRS

    hits   = 0
    report = []

    for pair in pairs:
        results = vector_store.similarity_search(pair["question"], k=top_k)
        combined = " ".join(
            [clean_text(doc.page_content).lower() for doc in results]
        )
        found_kw = [kw for kw in pair["keywords"] if kw.lower() in combined]
        hit      = len(found_kw) > 0
        if hit:
            hits += 1
        report.append({
            "question":      pair["question"],
            "hit":           hit,
            "found_keywords": found_kw,
            "all_keywords":   pair["keywords"],
        })

    hit_rate = hits / len(pairs) if pairs else 0
    return {"hit_rate": hit_rate, "total": len(pairs), "hits": hits, "report": report}


# ── Session state initialisation ──────────────────────────────────────────────
def init_session():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   # list of {"question": ..., "answer": ...}


# ── App ───────────────────────────────────────────────────────────────────────
setup_folders()
init_session()

st.set_page_config(page_title="PDF Study Assistant", page_icon="📚", layout="wide")
st.title("📚 PDF Study Assistant")
st.caption("Upload lecture PDFs, ask questions, and get answers grounded in your documents.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        for f in uploaded_files:
            with open(os.path.join(DOCUMENTS_FOLDER, f.name), "wb") as out:
                out.write(f.getbuffer())
        st.success(f"Uploaded {len(uploaded_files)} file(s).")

    st.divider()
    st.header("Retrieval settings")
    top_k        = st.slider("Chunks to retrieve (k)", 1, 10, 4)
    chunk_size   = st.slider("Chunk size (tokens)", 300, 1200, 700, step=50)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 120, step=20)

    st.divider()
    if st.button("🔨 Build / Rebuild database", use_container_width=True):
        with st.spinner("Chunking PDFs and building FAISS index…"):
            vs = create_vector_store(chunk_size, chunk_overlap)
        if vs:
            st.success("Database ready.")

    st.divider()
    use_llm = st.toggle("Use Ollama Llama 3 (local LLM)", value=False)
    if use_llm:
        st.caption("Make sure `ollama serve` is running with the `llama3` model pulled.")

    st.divider()
    st.header("Indexed PDFs")
    pdfs = [f for f in os.listdir(DOCUMENTS_FOLDER) if f.lower().endswith(".pdf")]
    if pdfs:
        for p in pdfs:
            st.write(f"📄 {p}")
    else:
        st.caption("No PDFs yet.")

    if st.button("🗑️ Clear all PDFs & index", use_container_width=True):
        shutil.rmtree(DOCUMENTS_FOLDER, ignore_errors=True)
        shutil.rmtree(VECTOR_FOLDER,    ignore_errors=True)
        setup_folders()
        st.session_state.chat_history = []
        st.success("Cleared.")
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation"])

# ── Chat tab ──────────────────────────────────────────────────────────────────
with tab_chat:
    # Render conversation history
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])

    # Input
    user_input = st.chat_input("Ask a question about your PDFs…")

    # Example questions (shown only when history is empty)
    if not st.session_state.chat_history:
        examples = [
            "Summarize the main topics covered in the PDFs.",
            "What are the key concepts explained in these documents?",
            "Generate 5 exam questions from the PDFs.",
            "Explain the most important idea from the uploaded material.",
        ]
        st.write("**Example questions:**")
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            if cols[i % 2].button(ex, key=f"ex_{i}"):
                user_input = ex

    if user_input:
        vs = load_vector_store()
        if vs is None:
            st.warning("Please build the PDF database first (sidebar → Build database).")
        else:
            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Searching PDFs…"):
                    reply = answer_question(
                        vs,
                        user_input,
                        st.session_state.chat_history,
                        use_llm,
                        top_k,
                    )
                st.markdown(reply)

            # Save turn to history
            st.session_state.chat_history.append(
                {"question": user_input, "answer": reply}
            )

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()

# ── Evaluation tab ────────────────────────────────────────────────────────────
with tab_eval:
    st.subheader("Retrieval quality check")
    st.write(
        "This runs a set of known question-keyword pairs against your index "
        "and checks whether relevant chunks are retrieved. "
        "A higher hit-rate means your chunking and embeddings are working well."
    )

    st.info("Default eval pairs are built-in. Questions are based on general document retrieval.")

    if st.button("▶️ Run evaluation", use_container_width=True):
        vs = load_vector_store()
        if vs is None:
            st.warning("Build the database first.")
        else:
            with st.spinner("Evaluating…"):
                results = run_evaluation(vs, top_k)

            hit_rate_pct = round(results["hit_rate"] * 100)
            col1, col2, col3 = st.columns(3)
            col1.metric("Hit rate",     f"{hit_rate_pct}%")
            col2.metric("Questions",    results["total"])
            col3.metric("Hits",         results["hits"])

            st.divider()
            for row in results["report"]:
                icon = "✅" if row["hit"] else "❌"
                with st.expander(f"{icon} {row['question']}"):
                    if row["hit"]:
                        st.success(f"Found keywords: {', '.join(row['found_keywords'])}")
                    else:
                        st.error(
                            f"None of the expected keywords found: "
                            f"{', '.join(row['all_keywords'])}"
                        )
                    st.caption(
                        "Tip: if this keeps failing, try reducing chunk size "
                        "or increasing k in the sidebar."
                    )
