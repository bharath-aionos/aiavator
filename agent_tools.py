import os
import sys,aiohttp,asyncio,sqlite3
from dotenv import load_dotenv
from loguru import logger
from datetime import datetime,timezone
import time,uuid,logging
from typing import Dict, Optional, List
import re

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.processors.frame_processor import FrameProcessor,FrameDirection
from pipecat.observers.base_observer import BaseObserver,FramePushed
from pipecat.frames.frames import (
   AudioRawFrame,
   Frame,
   TextFrame,
   EndFrame,
   StartFrame,
   TranscriptionFrame,
   LLMFullResponseStartFrame,
   LLMFullResponseEndFrame,
   TTSStartedFrame,
   TTSStoppedFrame,
   TTSAudioRawFrame,
   MetricsFrame,
   BotStartedSpeakingFrame,
   BotStoppedSpeakingFrame
)
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transcriptions.language import Language
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.frames.frames import TranscriptionMessage
from textblob import TextBlob
#from shared_state import live_transcriptions
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# RAG dependencies (aligned with RAG_UI.py; splitter is now in langchain_text_splitters)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import math

load_dotenv(override=True)



SYSTEM_INSTRUCTION = f"""
You are ITQ Guru, a friendly and helpful voice assistant.Your name is Manisha.
Your job is to answer user questions in a clear and concise way.

You have access to a single tool:
rag_search — Use this whenever a user asks a question. It will fetch relevant information from the knowledge base stored in a vector database.

Your responses should always:
Be short and to the point (one or two sentences).
Stay friendly, polite, and easy to understand.
Only use the information returned from the knowledge base.

IMPORTANT: When a user asks a question that might be in your knowledge base, you MUST use the rag_search function first before providing any answer. Always search the knowledge base for relevant information.

If no useful answer is found, say: "I’m sorry, I don’t have that information right now. Please try again later."

Do not ask for personal details or complicate the conversation.
Your goal is to clearly answer questions based on the knowledge base.
"""

# RAG globals and helpers (integrated, no Streamlit)
pdf_path = "travelport-smartpoint.pdf"  # ensure this file exists or adjust path
vectorstore = None
embedder = None
rag_chain = None

def split_into_factual_points(answer_text: str):
    """
    Simple sentence splitting into factual points. Keeps punctuation.
    """
    sents = re.split(r'(?<=[\.?\!])\s+', answer_text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    return sents

def keyword_overlap(a: str, b: str):
    wa = set(re.findall(r'\w+', a.lower()))
    wb = set(re.findall(r'\w+', b.lower()))
    if not wa:
        return 0.0
    return len(wa & wb) / len(wa)

def semantic_sim_score(point: str, doc_text: str, embedder):
    """
    Cosine similarity using embedding model (embedding on the fly for small sets).
    """
    v_point = embedder.embed_query(point)
    v_doc = embedder.embed_documents([doc_text])[0]
    dot = sum(a * b for a, b in zip(v_point, v_doc))
    norm_p = math.sqrt(sum(a * a for a in v_point))
    norm_d = math.sqrt(sum(a * a for a in v_doc))
    if norm_p == 0 or norm_d == 0:
        return 0.0
    return dot / (norm_p * norm_d)

def citefix_correct_answer(raw_answer: str, candidate_docs, embedder, lam=0.8, top_k=1):
    """
    Apply Keyword + Semantic Context correction per factual point.
    Returns corrected_text and pages used.
    """
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
        chosen = [d for s, d in scores[:Ci]]
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

def build_rag_index():
    global vectorstore, embedder, rag_chain
    if vectorstore is not None and rag_chain is not None and embedder is not None:
        return
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    for i, p in enumerate(pages):
        p.metadata["page"] = p.metadata.get("page", i)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedder)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
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
        chain_type_kwargs={"prompt": PROMPT},
    )

async def rag_search_handler(params:FunctionCallParams):
    """
    Handler for RAG search function calls
    """
    logger.info(f"[TOOL] rag_search called with query: {params.arguments}")
    try:
        query = params.arguments.get("query","")

        if not query:
            logger.warning("⚠️ [RAG_HANDLER] Empty query received")
            await params.result_callback("I need a search query to help you")
            return

        logger.info(f"🔎 [RAG_HANDLER] Processing query via backend: '{query}'")

        async with aiohttp.ClientSession() as session:
            async with session.post("http://localhost:8000/rag/query", json={"query": query}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    answer = data.get("answer") or "The information is not available in the document."
                    await params.result_callback(answer)
                else:
                    logger.error(f"[RAG_HANDLER] Backend returned status {resp.status}")
                    await params.result_callback("I'm having trouble finding that information. Please try again.")
    except Exception as e:
        logger.exception(f"[TOOL] Error in rag_search: {e}")
        await params.result_callback("I'm having trouble finding that information. Please try again.")


   # await params.result_callback("This is a test answer from the RAG tool.")

async def flight_search_handler(params:FunctionCallParams):
    logger.info(f"[TOOL] flight_search called with args: {params.arguments}")
    try:
        source = params.arguments.get("source", "")
        destination = params.arguments.get("destination", "")
        date = params.arguments.get("date", "")  # expected in DD-MM-YYYY

        if not source or not destination or not date:
            await params.result_callback("Please specify both source and destination cities and date.")
            return
        async with aiohttp.ClientSession() as session:
            payload = {"source": source, "destination": destination,"date": date}
            async with session.post("http://localhost:8000/search-flights", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    flights = data.get("flights", [])

                    if flights:
                        flight = flights[0]  # You can also loop through multiple
                        msg = (
                            f"Flight {flight['flight_number']} is a {flight['flight_type']} departing at {flight['departure_time']} "
                            f"and arriving at {flight['arrival_time']}. Current status is {flight['status']}. "
                            f"Economy seats are available for {int(flight['economy_price'])} dollars."
                        )
                        await params.result_callback(msg)
                    else:
                        await params.result_callback("Sorry, no flights found for your route.")
                else:
                    await params.result_callback("Flight search failed. Please try again.")
    except Exception as e:
        logger.exception(f"[TOOL] Error in flight_search: {e}")
        await params.result_callback("Something went wrong while searching for flights.")

async def pnr_search_handler(params: FunctionCallParams):
    logger.info(f"[TOOL] pnr_search called with args: {params.arguments}")
    try:
        pnr = params.arguments.get("pnr", "").strip().upper()

        if not pnr:
            await params.result_callback("Please provide your PNR number to continue.")
            return

        async with aiohttp.ClientSession() as session:
            async with session.post("http://localhost:8000/pnr-status", json={"pnr": pnr}) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "found":
                        msg = (
                            f"PNR {data['pnr_number']} belongs to {data['first_name']} {data['last_name']}. "
                            f"Flight {data['flight_number']} from {data['source']} to {data['destination']} on {data['date']}, "
                            f"departs at {data['departure_time']} and arrives at {data['arrival_time']}. "
                            f"Status: {data['booking_status']}."
                        )
                        await params.result_callback(msg)
                    else:
                        await params.result_callback(f"No booking found for PNR {pnr}. Please check and try again.")
                else:
                    await params.result_callback("Unable to check PNR right now. Please try again later.")
    except Exception as e:
        logger.exception("[TOOL] Error in pnr_search")
        await params.result_callback("Something went wrong while checking your PNR. Please try again.")


async def run_bot2(webrtc_connection):

    pipecat_transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            audio_out_10ms_chunks=2,
        ),
    )
    logger.debug("🔌 [BOT] Transport initialized")


    async with aiohttp.ClientSession() as session:
        stt = GroqSTTService(api_key=os.getenv("GROQ_API_KEY"),model="whisper-large-v3-turbo")
        tts = SarvamTTSService(api_key=os.getenv("SARVAM_API_KEY"),voice_id="manisha",model="bulbul:v2",aiohttp_session=session,params=SarvamTTSService.InputParams(language=Language.EN))
        llm = GroqLLMService(api_key=os.getenv("GROQ_API_KEY"),model="meta-llama/llama-4-maverick-17b-128e-instruct")
        logger.debug("🤖 [BOT] LLM service initialized")

        llm.register_function("rag_search",rag_search_handler)
        llm.register_function("flight_search", flight_search_handler)
        llm.register_function("pnr_search", pnr_search_handler)


        logger.info(" [BOT] RAG search function registered with LLM")

        rag_function = FunctionSchema(
            name="rag_search",
            description= "Search the knowledge base for information about ",
            properties={
                "query":{
                    "type":"string",
                    "description":"The user's question or search query",
                }
            },
            required=["query"]
        )

        flight_function = FunctionSchema(
            name="flight_search",
            description="Search for flights between two cities",
            properties={
                "source": {
                    "type": "string",
                    "description": "Departure city"
                },
                "destination": {
                    "type": "string",
                    "description": "Arrival city"
                },
                "date":{
                    "type": "string",
                    "description": "Date of travel in DD-MM-YYYY format"
                }
            },
            required=["source", "destination","date"]
        )

        pnr_function = FunctionSchema(
            name="pnr_search",
            description="Check the booking status using a PNR number",
            properties={
                "pnr": {
                    "type": "string",
                    "description": "The user's 6-8 character PNR number"
                }
            },
            required=["pnr"]
        )



        tools = ToolsSchema(standard_tools=[rag_function])

        messages = [
            {
                "role":"system",
                "content":SYSTEM_INSTRUCTION,
            },
        ]

        context = OpenAILLMContext(messages=messages,tools=tools)
        logger.debug("📝 [BOT] Context initialized with tools")
        context_aggregator = llm.create_context_aggregator(context)
        logger.debug("📊 [BOT] Context aggregator created")

        #session_id = str(uuid.uuid4())
        #latency_tracker = LatencyTracker(session_id)

        pipeline = Pipeline(
            [
                pipecat_transport.input(),
                stt,
                context_aggregator.user(),
                llm,  # LLM
                tts,
                pipecat_transport.output(),
                context_aggregator.assistant(),
                #latency_tracker
            ]
        )


        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
                allow_interruptions=True,
            ),
        )

        @pipecat_transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Pipecat Client connected")
            # Kick off the conversation.
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @pipecat_transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Pipecat Client disconnected")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=False)

        await runner.run(task)