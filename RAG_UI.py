# app_citefix.py
import os
import re
import fitz
import streamlit as st
from PIL import Image

# LangChain / embeddings / vectorstore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------- Config ----------
st.set_page_config(page_title="📘 PDF Chat + CiteFix", layout="wide")
st.title("📘 PDF Chatbot + CiteFix (Keyword + Semantic Citation Correction)")

# Replace with your Groq key (or use streamlit secrets)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def render_page_image(pdf_path: str, page_number: int):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

@st.cache_resource(show_spinner=True)
def build_index(pdf_path: str, chunk_size=800, chunk_overlap=150):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    # ensure page metadata is present (PyPDFLoader may include it; set explicitly)
    for i, p in enumerate(pages):
        # store 1-based page in metadata for clarity later
        p.metadata["page"] = p.metadata.get("page", i)  # zero-based, we'll +1 when display

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(pages)

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedder)
    return vectorstore, embedder, len(pages), len(chunks)

def split_into_factual_points(answer_text: str):
    """
    Simple sentence splitting into factual points. Keeps punctuation.
    """
    # split on sentence boundaries (., ?, !), keep the delimiter
    sents = re.split(r'(?<=[\.\?\!])\s+', answer_text.strip())
    # filter empty
    sents = [s.strip() for s in sents if s.strip()]
    return sents

def keyword_overlap(a: str, b: str):
    wa = set(re.findall(r'\w+', a.lower()))
    wb = set(re.findall(r'\w+', b.lower()))
    if not wa: return 0.0
    return len(wa & wb) / len(wa)

def semantic_sim_score(point: str, doc_text: str, embedder):
    """
    Compute cosine similarity using embedding model (embedding both on the fly).
    For small numbers of documents this is OK in latency.
    """
    # embeddings from HuggingFaceEmbeddings return list of floats; compute cosine
    v_point = embedder.embed_query(point)
    v_doc = embedder.embed_documents([doc_text])[0]
    # cosine
    import math
    dot = sum(a*b for a,b in zip(v_point, v_doc))
    norm_p = math.sqrt(sum(a*a for a in v_point))
    norm_d = math.sqrt(sum(a*a for a in v_doc))
    if norm_p == 0 or norm_d == 0:
        return 0.0
    return dot / (norm_p * norm_d)

def citefix_correct_answer(raw_answer: str, candidate_docs, embedder, lam=0.8, top_k=1):
    """
    Apply Keyword + Semantic Context correction per factual point.
    candidate_docs: list of Document (with metadata and page and page_content)
    Returns corrected_answer_text and set of pages used.
    """
    points = split_into_factual_points(raw_answer)
    corrected_points = []
    pages_used = []
    for pt in points:
        # count how many citations LLM already placed in this point (naive: occurrences of "Page X" or "Source")
        found = re.findall(r'page\s*(\d+)', pt, flags=re.IGNORECASE)
        Ci = len(found) if found else 1

        # score each candidate doc
        scores = []
        for doc in candidate_docs:
            kw = keyword_overlap(pt, doc.page_content)
            sem = semantic_sim_score(pt, doc.page_content, embedder)
            score = lam * kw + (1 - lam) * sem
            scores.append((score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)
        # choose top Ci docs
        chosen = [d for s,d in scores[:Ci]]
        # attach citations to the factual point
        if chosen:
            # choose first as primary
            primary = chosen[0]
            page_meta = primary.metadata.get("page", 0)
            page_one_based = page_meta + 1  # because earlier we stored 0-based
            pages_used.append(page_one_based)
            corrected_pt = f"{pt} (Source: Page {page_one_based})"
        else:
            corrected_pt = pt  # leave as-is if nothing found
        corrected_points.append(corrected_pt)

    final = " ".join(corrected_points)
    return final, sorted(set(pages_used))

# ---------- App UI ----------
col1, col2 = st.columns([0.6, 0.4])

with col1:
    st.header("Upload & Process PDF")
    uploaded = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded:
        temp_dir = "temp_pdf_files"
        os.makedirs(temp_dir, exist_ok=True)
        pdf_path = os.path.join(temp_dir, uploaded.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved {uploaded.name}")
        st.session_state['pdf_path'] = pdf_path

    # Process button
    if st.button("Process Document") and st.session_state.get('pdf_path'):
        path = st.session_state['pdf_path']
        with st.spinner("Building index..."):
            vectorstore, embedder, num_pages, num_chunks = build_index(path)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
            # LLM and prompt: ask for citations in the LLM output (helps segmentation)
            llm = ChatGroq(model="llama-3.1-8b-instant")
            prompt_template = """You are an assistant answering using only the provided context.
Please append citation markers after factual statements in the form " (Page N)". If the fact is not in the context, say "The information is not available in the document."

Context:
{context}

Question:
{question}

Answer:"""
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            # store in session
            st.session_state['vectorstore'] = vectorstore
            st.session_state['embedder'] = embedder
            st.session_state['retriever'] = retriever
            st.session_state['rag_chain'] = rag_chain
            st.session_state['pdf_processed'] = True
            st.session_state['pdf_pages'] = num_pages
            st.success("Index built. You can ask questions now.")
            st.info(f"Pages: {num_pages} | Chunks: {num_chunks}")

    # Query / Chat
    if st.session_state.get('pdf_processed'):
        st.markdown("---")
        st.header("Ask a question")
        query = st.text_input("Enter your question here and press Enter")
        if query:
            with st.spinner("Retrieving and generating..."):
                rag_chain = st.session_state['rag_chain']
                # get raw result (LLM will see retrieved context)
                response = rag_chain.invoke({"query": query})
                # 'result' or 'answer' depending on version
                raw_answer = response.get("result") or response.get("answer") or ""
                # candidate docs from retriever (we want the docs the retriever returned)
                candidate_docs = response.get("source_documents") or st.session_state['retriever'].get_relevant_documents(query)
                # run CiteFix correction
                corrected_answer, pages_used = citefix_correct_answer(raw_answer, candidate_docs, st.session_state['embedder'], lam=0.8, top_k=1)

                # display
                st.subheader("Corrected Answer (CiteFix applied)")
                st.write(corrected_answer)

                # detailed block: show which original retrieved docs were used
                st.markdown("**Candidate documents (retriever)**")
                for i, d in enumerate(candidate_docs[:6]):
                    p = d.metadata.get("page", 0) + 1
                    src = d.metadata.get("source", st.session_state.get('pdf_path'))
                    st.markdown(f"- Candidate {i+1}: Page {p} — snippet: {d.page_content[:180].strip()}...")

                # Save source pages to session for right-side viewer
                st.session_state['source_pages'] = pages_used

with col2:
    st.header("Relevant Source Pages (CiteFix)")
    if st.session_state.get('source_pages'):
        pages = st.session_state['source_pages']
        # display in columns
        cols = st.columns(len(pages))
        for idx, pg in enumerate(pages):
            try:
                img = render_page_image(st.session_state['pdf_path'], pg)
                with cols[idx]:
                    st.image(img, caption=f"Page {pg}", use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render page {pg}: {e}")
    else:
        st.info("After asking a question, corrected source pages will appear here.")
