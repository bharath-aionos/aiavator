import argparse,os,re,logging,sqlite3,base64,io
import asyncio
import sys
import pandas as pd
from contextlib import asynccontextmanager
from typing import Dict
from datetime import datetime
import uvicorn
from agent_tools import run_bot2
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request,WebSocket, UploadFile, File, HTTPException
from fastapi.responses import FileResponse,JSONResponse,Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import fitz
from PIL import Image
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# RAG dependencies (mirroring RAG_UI.py, no Streamlit)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection

# Load environment variables
load_dotenv(override=True)

# -------- RAG (FAISS/LangChain) globals & helpers (no Streamlit) --------
# Stores last processed PDF, vector index and chain
RAG_STATE = {
    "pdf_path": None,
    "vectorstore": None,
    "embedder": None,
    "rag_chain": None,
    "num_pages": 0,
    "num_chunks": 0,
}

# Last RAG result cache for UI/voice assistant to retrieve
LAST_RAG_RESULT = {
    "answer": None,
    "pages": [],
    "timestamp": None,
}

def split_into_factual_points(answer_text: str):
    sents = re.split(r'(?<=[\.?\!])\s+', (answer_text or "").strip())
    return [s.strip() for s in sents if s.strip()]

def keyword_overlap(a: str, b: str):
    wa = set(re.findall(r'\w+', (a or "").lower()))
    wb = set(re.findall(r'\w+', (b or "").lower()))
    if not wa:
        return 0.0
    return len(wa & wb) / len(wa)

def semantic_sim_score(point: str, doc_text: str, embedder):
    import math
    v_point = embedder.embed_query(point)
    v_doc = embedder.embed_documents([doc_text])[0]
    dot = sum(a*b for a,b in zip(v_point, v_doc))
    norm_p = math.sqrt(sum(a*a for a in v_point))
    norm_d = math.sqrt(sum(a*a for a in v_doc))
    if norm_p == 0 or norm_d == 0:
        return 0.0
    return dot / (norm_p * norm_d)

def citefix_correct_answer(raw_answer: str, candidate_docs, embedder, lam=0.8):
    points = split_into_factual_points(raw_answer)
    corrected_points = []
    pages_used = []
    for pt in points:
        found = re.findall(r'page\s*(\d+)', pt, flags=re.IGNORECASE)
        Ci = len(found) if found else 1
        scores = []
        for doc in candidate_docs or []:
            kw = keyword_overlap(pt, doc.page_content)
            sem = semantic_sim_score(pt, doc.page_content, embedder)
            score = lam * kw + (1 - lam) * sem
            scores.append((score, doc))
        scores.sort(key=lambda x: x[0], reverse=True)
        chosen = [d for s,d in scores[:Ci]]
        if chosen:
            primary = chosen[0]
            page_meta = primary.metadata.get("page", 0)
            page_one_based = page_meta + 1
            pages_used.append(page_one_based)
            corrected_pt = f"{pt} (Source: Page {page_one_based})"
        else:
            corrected_pt = pt
        corrected_points.append(corrected_pt)
    final = " ".join(corrected_points)
    return final, sorted(set(pages_used))

def build_rag_index(pdf_path: str, chunk_size=1200, chunk_overlap=250):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    for i, p in enumerate(pages):
        p.metadata["page"] = p.metadata.get("page", i)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(pages)
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedder)
    # Use MMR retriever to improve diversity and relevance, fetch more then select top-k
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 12, "fetch_k": 60, "lambda_mult": 0.5})
    # Lower temperature for deterministic answers grounded in context, stronger model for better grounding
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
    prompt_template = """You are a careful assistant. Answer using ONLY the provided context.
Write a clear, complete answer that directly addresses the question.
Prefer specifics over generic statements. Do not invent facts.

Rules:
- After any factual sentence derived from the context, append a citation in the form " (Page N)".
- If the answer is not present in the context, reply: "The information is not available in the document."
- If there are multiple relevant points, summarize them succinctly in 2-5 sentences.

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
    return vectorstore, embedder, rag_chain, len(pages), len(chunks)

def render_page_image(pdf_path: str, page_number: int):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

api_key = os.getenv("PINECONE_API_KEY")
environment = "us-east-1"
index_name = "travelport-smartpoint"

pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)

# Embedding model
model = SentenceTransformer("intfloat/e5-base")

#FLIGHT_DATA = pd.read_csv("flight_status_data.csv")

# Store connections by pc_id
pcs_map: Dict[str, SmallWebRTCConnection] = {}

XIRSYS_USERNAME = os.getenv("XIRSYS_USERNAME")
XIRSYS_CREDENTIAL = os.getenv("XIRSYS_CREDENTIAL")

# Configure ICE servers using your Xirsys credentials
ice_servers = [
    # STUN server from Xirsys
    IceServer(
        urls="stun:bn-turn1.xirsys.com",
    ),
    # TURN servers from Xirsys (UDP, TCP, TLS over TCP)
    # The pipecat IceServer class expects a single URL string for 'urls'.
    # So, we'll create separate IceServer objects for each distinct URL you want to use.
    # It's good to include UDP, TCP, and TLS options for maximum compatibility.

    # UDP TURN servers
    IceServer(
        urls="turn:bn-turn1.xirsys.com:80?transport=udp",
        username=XIRSYS_USERNAME,
        credential=XIRSYS_CREDENTIAL,
    ),
    IceServer(
        urls="turn:bn-turn1.xirsys.com:3478?transport=udp",
        username=XIRSYS_USERNAME,
        credential=XIRSYS_CREDENTIAL,
    ),
    # TCP TURN servers
    IceServer(
        urls="turn:bn-turn1.xirsys.com:80?transport=tcp",
        username=XIRSYS_USERNAME,
        credential=XIRSYS_CREDENTIAL,
    ),
    IceServer(
        urls="turn:bn-turn1.xirsys.com:3478?transport=tcp",
        username=XIRSYS_USERNAME,
        credential=XIRSYS_CREDENTIAL,
    ),
    # TLS over TCP TURN servers (often the most reliable for strict firewalls)
    IceServer(
        urls="turns:bn-turn1.xirsys.com:443?transport=tcp",
        username=XIRSYS_USERNAME,
        credential=XIRSYS_CREDENTIAL,
    ),
    IceServer(
        urls="turns:bn-turn1.xirsys.com:5349?transport=tcp",
        username=XIRSYS_USERNAME,
        credential=XIRSYS_CREDENTIAL,
    ),
]

@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(sdp=request["sdp"], type=request["type"])
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        background_tasks.add_task(run_bot2, pipecat_connection)

    answer = pipecat_connection.get_answer()
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


@app.post("/login")
async def login(request: Request):
    body = await request.json()
    username = body.get("username")
    password = body.get("password")
    if username == "admin" and password == "password":
        return {"success": True}
    else:
        return {"success": False, "error": "Invalid credentials"}

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

@app.post("/pdf/upload")
async def pdf_upload(file: UploadFile = File(...)):
    try:
        os.makedirs("temp_pdf_files", exist_ok=True)
        pdf_path = os.path.join("temp_pdf_files", file.filename)
        with open(pdf_path, "wb") as f:
            f.write(await file.read())
        RAG_STATE["pdf_path"] = pdf_path
        logger.success(f"📄 [PDF] Uploaded and saved to {pdf_path}")
        return {"pdf_path": pdf_path, "filename": file.filename}
    except Exception:
        logger.exception("[PDF] Upload failed")
        raise HTTPException(status_code=500, detail="Upload failed")

@app.post("/pdf/process")
async def pdf_process(request: Request):
    body = await request.json()
    pdf_path = body.get("pdf_path") or RAG_STATE.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=400, detail="PDF path not set or file missing")
    try:
        vectorstore, embedder, rag_chain, num_pages, num_chunks = build_rag_index(pdf_path)
        RAG_STATE.update({
            "vectorstore": vectorstore,
            "embedder": embedder,
            "rag_chain": rag_chain,
            "num_pages": num_pages,
            "num_chunks": num_chunks,
        })
        logger.success(f"🧭 [RAG] Index built: pages={num_pages}, chunks={num_chunks}")
        return {"pages": num_pages, "chunks": num_chunks}
    except Exception:
        logger.exception("[RAG] Failed to build index")
        raise HTTPException(status_code=500, detail="Failed to process PDF")

@app.post("/rag/query")
async def rag_query(request: Request):
    if not RAG_STATE.get("rag_chain") or not RAG_STATE.get("embedder"):
        raise HTTPException(status_code=400, detail="RAG index not built. Upload and process a PDF first.")
    body = await request.json()
    query = body.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query'")
    try:
        rag_chain = RAG_STATE["rag_chain"]
        response = rag_chain.invoke({"query": query})
        raw_answer = response.get("result") or response.get("answer") or ""
        candidate_docs = response.get("source_documents")
        if not candidate_docs and RAG_STATE.get("vectorstore"):
            try:
                fallback = RAG_STATE["vectorstore"].as_retriever(search_kwargs={"k": 12})
                candidate_docs = fallback.get_relevant_documents(query)
            except Exception:
                candidate_docs = []
        corrected_answer, pages_used = citefix_correct_answer(raw_answer, candidate_docs, RAG_STATE["embedder"], lam=0.5)
        # Relevance guard: if nothing cited and retrieved docs are weak vs query, report not available
        try:
            sims = []
            for d in (candidate_docs or [])[:6]:
                sims.append(semantic_sim_score(query, d.page_content or "", RAG_STATE["embedder"]))
            max_sim = max(sims) if sims else 0.0
            if (not pages_used) and max_sim < 0.35:
                corrected_answer = "The information is not available in the document."
        except Exception:
            pass
        # pack candidate snippets
        candidates = []
        for d in (candidate_docs or [])[:6]:
            p = (d.metadata.get("page", 0) + 1)
            candidates.append({"page": p, "snippet": (d.page_content or "")[:200]})
        # save last result for UI/voice display
        try:
            from datetime import datetime as _dt
            LAST_RAG_RESULT.update({
                "answer": corrected_answer,
                "pages": pages_used,
                "timestamp": _dt.utcnow().isoformat() + "Z",
            })
        except Exception:
            pass
        return {"answer": corrected_answer, "pages": pages_used, "candidates": candidates}
    except Exception:
        logger.exception("[RAG] Query failed")
        raise HTTPException(status_code=500, detail="Query failed")

@app.get("/rag/last")
async def rag_last():
    return LAST_RAG_RESULT

@app.get("/pdf/page/{page}")
async def pdf_page(page: int):
    pdf_path = RAG_STATE.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=400, detail="No PDF loaded")
    if page < 1 or (RAG_STATE.get("num_pages") and page > RAG_STATE["num_pages"]):
        raise HTTPException(status_code=400, detail="Page out of range")
    try:
        buf = render_page_image(pdf_path, page)
        return StreamingResponse(buf, media_type="image/png")
    except Exception:
        logger.exception("[PDF] Render failed")
        raise HTTPException(status_code=500, detail="Render failed")

def get_embedding(text: str):
    logger.debug(f"🔢 [EMBEDDING] Generating embedding for text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    embedding = model.encode(text).tolist()
    logger.debug(f"✅ [EMBEDDING] Generated embedding with {len(embedding)} dimensions")
    return embedding

def clean_answer_for_voice(answer: str) -> str:
    """
    Clean FAQ answer for voice output by removing special characters
    and making it more conversational
    """
    # Remove common symbols and replace with voice-friendly alternatives
    replacements = {
        '&': ' and ',
        '%': ' percent ',
        '#': ' hashtag ',
        '@': ' at the rate ',
        '*': '',
        '•': '',
        '₹': ' rupees ',
        '$': ' dollars ',
        '€': ' euros ',
        '£': ' pounds ',
    }
    
    cleaned = answer
    for symbol, replacement in replacements.items():
        cleaned = cleaned.replace(symbol, replacement)
    
    # Remove extra whitespace
    cleaned = ' '.join(cleaned.split())
    
    # Limit length for voice output (roughly 15-20 seconds of speech)
    words = cleaned.split()
    if len(words) > 50:  # Roughly 15-20 seconds at normal speaking pace
        cleaned = ' '.join(words[:50]) + '...'
    
    return cleaned

@app.post("/rag")
async def rag_search(request: Request):

    logger.info("🔍 [RAG_ENDPOINT] RAG search request received")
    try:
        body = await request.json()
        user_query = body.get("query")

        logger.info(f"📝 [RAG_ENDPOINT] Query extracted: '{user_query}'")

        if not user_query:
            logger.warning("⚠️ [RAG_ENDPOINT] Missing 'query' in request body")
            return {"error": "Missing 'query' in request body"}

        logger.info(f"Processing RAG for query: {user_query}")

        query_vector = get_embedding(user_query)
        logger.debug(f"🔢 [RAG_ENDPOINT] Generated query vector with {len(query_vector)} dimensions")
        logger.info("🔍 [RAG_ENDPOINT] Querying Pinecone index")
        #results = index.query(vector=query_vector, top_k=3, include_metadata=True)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: index.query(
                vector=query_vector,
                top_k=3,
                include_metadata=True
            )
        )
        logger.debug(f"📊 [RAG_ENDPOINT] Pinecone query completed")

        matches = results.get("matches", [])
        logger.info(f"🎯 [RAG_ENDPOINT] Found {len(matches)} matches")

        if matches:
            # Log all matches with scores
            #for i, match in enumerate(matches):
            #    score = match.get("score", 0)
            #    question = match.get("metadata", {}).get("question", "N/A")
            #    logger.debug(f"📋 [RAG_ENDPOINT] Match {i+1}: Score={score:.4f}, Question='{question[:100]}{'...' if len(question) > 100 else ''}'")

            best_match = matches[0]
            best_score = best_match.get("score", 0)
            logger.info(f"🏆 [RAG_ENDPOINT] Best match score: {best_score:.4f}")

            if best_score > 0.80:
                answer = best_match.get("metadata", {}).get("text", "")
                logger.success(f"✅ [RAG_ENDPOINT] High confidence match found. Raw answer: '{answer[:100]}{'...' if len(answer) > 100 else ''}'")
                
                if answer:
                    cleaned = clean_answer_for_voice(answer)
                    return {"answer": cleaned}
                else:
                    return {"answer": "I couldn’t find a proper answer."}
            else:
                logger.warning(f"⚠️ [RAG_ENDPOINT] Best match score {best_score:.4f} below threshold (0.80)")
        else:
            logger.warning("⚠️ [RAG_ENDPOINT] No matches found in Pinecone")
            logger.info("🤷 [RAG_ENDPOINT] Returning default response")
            return {"answer": "Let me check that for you."}
    except Exception as e:
        logger.exception("Error during RAG search")
        return {"error": "Internal server error"}

@app.post("/search-flights")
async def search_flights(request: Request):
    body = await request.json()
    source = body.get("source","").strip().lower()
    destination = body.get("destination","").strip().lower()
    date_str = body.get("date", "").strip()  # format expected: DD-MM-YYYY
    logger.info(f"✈️ [FLIGHT_SEARCH] Searching flights from '{source}' to '{destination}'")

    if not source or not destination or not date_str:
        return {"error":"Missing source or destination or date"}
    
    # Convert date to match CSV format
    try:
        date_obj = datetime.strptime(date_str, "%d-%m-%Y")
        formatted_date = date_obj.strftime("%Y-%m-%d")  # Matches CSV
    except ValueError:
        return {"error": "Invalid date format. Use DD-MM-YYYY."}
    
    logger.debug(f"🧾 [DEBUG] CSV Sources: {FLIGHT_DATA['Source'].tolist()}")
    logger.debug(f"🧾 [DEBUG] CSV Destinations: {FLIGHT_DATA['Destination'].tolist()}")
    logger.debug(f"🧾 [DEBUG] CSV Dates: {FLIGHT_DATA['Date'].tolist()}")
    
    logger.debug(f"🔁 [DEBUG] Matching against: source='{source}', destination='{destination}', date='{formatted_date}'")

    #filter flights
    filtered = FLIGHT_DATA[
        (FLIGHT_DATA["Source"].astype(str).str.strip().str.lower() == source) &
        (FLIGHT_DATA["Destination"].astype(str).str.strip().str.lower() == destination) &
        (FLIGHT_DATA["Date"].astype(str).str.strip() == formatted_date)
    ]
    logger.info(f"🔎 [FLIGHT_SEARCH] Matched {len(filtered)} flights")


    if not filtered.empty:
        results = []
        for _,row in filtered.iterrows():
            result = {
                "flight_type":row["Flight Type"],
                "flight_number": row["Flight Number"],
                "departure_time": row["Departure Time"],
                "arrival_time": row["Arrival Time"],
                "status": row["Status"],
                "economy_price": row["Price (Economy)"],
                "business_price": row["Price (Business)"]
            }
            results.append(result)

        logger.success(f"✅ [FLIGHT_SEARCH] Found {len(results)} flight(s)")
        return {"flights": results}
    else:
        logger.warning("❌ [FLIGHT_SEARCH] No flights found")
        return {"flights": []}

@app.post("/pnr-status")
async def pnr_status(request:Request):
    body = await request.json()
    pnr = body.get("pnr","").strip().upper()

    logger.info(f"📦 [PNR] Searching PNR: {pnr}")

    if not pnr:
        return {"error": "PNR number is required."}
    
    try:
        conn = sqlite3.connect("pnr_database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pnr_status WHERE pnr_number = ?", (pnr,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            logger.warning(f"❌ [PNR] No booking found for PNR: {pnr}")
            return {"status": "not_found"}

        logger.success(f"✅ [PNR] Booking found: {row}")
        return {
            "status": "found",
            "pnr_number": row[0],
            "first_name": row[1],
            "last_name": row[2],
            "flight_number": row[3],
            "source": row[4],
            "departure_time": row[5],
            "destination": row[6],
            "arrival_time": row[7],
            "date": row[8],
            "booking_status": row[9]
        }

    except Exception as e:
        logger.exception("[PNR] Error while checking PNR")
        return {"error": "Internal server error"}
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC demo")
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for HTTP server (default: 8000)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    logger.remove(0)
    if args.verbose:
        logger.add(sys.stderr, level="TRACE")
    else:
        logger.add(sys.stderr, level="DEBUG")

    uvicorn.run(app, host=args.host, port=args.port)