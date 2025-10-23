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

api_key = os.getenv("PINECONE_API_KEY")
index_name = "travelport-smartpoint"
pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)

model = SentenceTransformer("intfloat/e5-base")

def get_embedding(text: str):
    return model.encode(text).tolist()

async def rag_search_handler(params:FunctionCallParams):
    """
    Handler for RAG search function calls
    """
    logger.info(f"[TOOL] rag_searc
                h called with query: {params.arguments}")
    try:
        query = params.arguments.get("query","")

        if not query:
            logger.warning("⚠️ [RAG_HANDLER] Empty query received")
            await params.result_callback("I need a search query to help you")
            return

        logger.info(f"🔎 [RAG_HANDLER] Processing query: '{query}'")

        query_vector = get_embedding(query)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: index.query(vector=query_vector, top_k=3, include_metadata=True)
        )

        matches = results.get("matches", [])
        if matches:
            best_match = matches[0]
            score = best_match.get("score", 0)
            text_answer = best_match.get("metadata", {}).get("text", "")

            if score > 0.80 and text_answer:
                await params.result_callback(text_answer)
            else:
                await params.result_callback("Let me check that for you.")
        else:
            await params.result_callback("I’m sorry, I don’t have that information right now. Please try again later.")
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


        logger.info("🔧 [BOT] RAG search function registered with LLM")

        rag_function = FunctionSchema(
            name="rag_search",
            description= "Search the knowledge base for information about Indigo Voice services, account issues,faq questions and general company information",
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