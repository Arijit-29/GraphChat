from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Any, Dict, Optional
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command,interrupt
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests
import sqlite3
import os
import tempfile
import uuid

load_dotenv()
OPEN_WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "Chatbot"


llm = ChatGroq(model="openai/gpt-oss-120b", streaming=True, reasoning_effort="low", reasoning_format="parsed")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


search_tool = DuckDuckGoSearchRun(region="us-en")

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(
    file_bytes: bytes, thread_id: str, filename: Optional[str] = None
) -> dict:
    """
    Build a isolated Chroma retriever for the uploaded PDF and store it for the thread.
    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        safe_thread_id = str(thread_id).replace("-", "")
        unique_collection_name = f"thread_{safe_thread_id}_{uuid.uuid4().hex[:8]}"

        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            collection_name=unique_collection_name  # <- This forces data isolation
        )
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


@tool
def rag_tool(query: str, config: RunnableConfig) -> str:
    """
    Search the uploaded PDF document for relevant information.
    Always use this tool when the user asks about the document, summarizing, or extracting facts.
    """
    thread_id = config.get("configurable", {}).get("thread_id")
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return "Error: No PDF document is attached to this chat. Ask the user to upload a PDF first."
    result = retriever.invoke(query)
    if not result:
        return f"No relevant information found in the document for the query: '{query}'"
    formatted_context = f"--- Document Results for '{query}' ---\n\n"
    for i, doc in enumerate(result, 1):
        clean_text = " ".join(doc.page_content.split())
        formatted_context += f"[Excerpt {i}]: {clean_text}\n\n"

    return formatted_context


@tool
def calculator(first_num: float, second_num: float, operation: str = "add") -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        operation = (operation or "add").lower()
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    r = requests.get(url)
    return r.json()


@tool
def get_weather_data(city: str) -> dict:
    """Get 2-day weather forecast for a city."""
    url = f"https://api.weatherapi.com/v1/forecast.json?key={OPEN_WEATHER_API_KEY}&q={city}&days=2"
    data = requests.get(url).json()
    return {
        "city": data["location"]["name"],
        "country": data["location"]["country"],
        "forecast": [
            {
                "date": day["date"],
                "max_temp": day["day"]["maxtemp_c"],
                "min_temp": day["day"]["mintemp_c"],
                "condition": day["day"]["condition"]["text"],
            }
            for day in data["forecast"]["forecastday"]
        ],
    }


tools = [search_tool, get_stock_price, calculator, get_weather_data, rag_tool]
llm_with_tools = llm.bind_tools(tools,parallel_tool_calls=False)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState, config:RunnableConfig):
    """LLM node that may answer or request a tool call."""
    system_message = SystemMessage(
        content=(
           "You are a highly capable AI assistant.\n"
            "CRITICAL INSTRUCTION: If the user asks ANY question about the uploaded document, "
            "you MUST immediately use the `rag_tool` to search for the answer. "
            "Do NOT rely on your general knowledge. Base your answer strictly on the tool's text excerpts. "
            "You also have access to web search, stock prices, and a calculator."
        )
    )
    messages = [system_message] + state["messages"]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)
# Checkpointer
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")


chatbot = graph.compile(checkpointer=checkpointer, interrupt_before=["tools"])


def delete_thread(thread_id: str):
    """Delete a persisted chat thread from the SQLite checkpoint store."""
    checkpointer.delete_thread(thread_id)


def retrieve_all_threads():
    threads = []
    seen = set()
    for checkpoint in checkpointer.list(None):
        tid = checkpoint.config["configurable"]["thread_id"]
        if tid not in seen:
            seen.add(tid)
            threads.append(tid)
    return threads


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})
