import os
import sys
import uuid
import streamlit as st
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage,
)

# Ensure backend can be imported
sys.path.insert(0, os.path.dirname(__file__))
from langgraph_backend import (
    chatbot,
    retrieve_all_threads,
    delete_thread,
    ingest_pdf,
    thread_document_metadata,
)

# ------------------------ Page Configuration --------------------------------#
st.set_page_config(
    page_title="GraphChat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------ State Management ----------------------------------#
def initialize_session():
    """Initialize core session variables required for routing."""
    if "user_id" in st.query_params:
        st.session_state.user_id = st.query_params["user_id"]
    if "user_id" not in st.session_state:
        new_id = str(uuid.uuid4())
        st.session_state.user_id = new_id
        st.query_params["user_id"] = new_id
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = (
            f"{st.session_state.user_id}_{uuid.uuid4().hex[:8]}"
        )
    if "chat_threads" not in st.session_state:
        st.session_state.chat_threads = retrieve_all_threads(st.session_state.user_id)
    if st.session_state.thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.insert(0, st.session_state.thread_id)
    if "ingested_docs" not in st.session_state:
        st.session_state["ingested_docs"] = {}


def get_thread_state(thread_id):
    """Fetch the source-of-truth message history directly from LangGraph."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def get_chat_title(messages):
    """Derive a clean title from the first human message."""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content[:30] + ("..." if len(content) > 30 else "")  # type: ignore
    return "New Conversation"


def start_new_chat():
    """Reset the active thread to a fresh UUID."""
    st.session_state.thread_id = f"{st.session_state.user_id}_{uuid.uuid4().hex[:8]}"
    st.session_state.chat_threads.insert(0, st.session_state.thread_id)


# ------------------------ Sidebar UI ---------------------------------------#
initialize_session()

with st.sidebar:
    st.title("💬 GraphChat")

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()

    st.divider()
    st.subheader("📄 Document Context")

    # Check if a document is already loaded for the current thread
    doc_meta = thread_document_metadata(st.session_state.thread_id)

    if doc_meta:
        st.success(f"Attached: {doc_meta.get('filename')}")
        st.caption(
            f"{doc_meta.get('documents')} pages | {doc_meta.get('chunks')} chunks indexed."
        )
    else:
        st.info("No document attached to this chat.")

    # Dynamic key ensures the uploader resets when switching threads
    uploaded_file = st.file_uploader(
        "Upload a PDF for RAG",
        type=["pdf"],
        key=f"uploader_{st.session_state.thread_id}",
    )

    if uploaded_file:
        # Only ingest if it's a new file (prevents re-ingesting on every Streamlit rerun)
        if not doc_meta or doc_meta.get("filename") != uploaded_file.name:
            with st.spinner("Processing and indexing PDF..."):
                try:
                    # Pass the raw bytes, thread_id, and filename to the backend
                    file_bytes = uploaded_file.read()
                    ingest_pdf(
                        file_bytes, st.session_state.thread_id, uploaded_file.name
                    )
                    st.rerun()  # Refresh the UI to show the success message and metadata
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")

    st.divider()
    st.subheader("📋 Recent Conversations")

    if not st.session_state.chat_threads:
        st.info("No conversations yet.")
    else:
        for thread_id in st.session_state.chat_threads:
            messages = get_thread_state(thread_id)
            title = get_chat_title(messages)

            is_active = thread_id == st.session_state.thread_id
            btn_label = f"📍 {title}" if is_active else f"💭 {title}"

            col1, col2 = st.columns([8, 2])
            with col1:
                if st.button(
                    btn_label, key=f"load_{thread_id}", use_container_width=True
                ):
                    st.session_state.thread_id = thread_id
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"del_{thread_id}", help="Delete chat"):
                    delete_thread(thread_id)
                    st.session_state.chat_threads = retrieve_all_threads(
                        st.session_state.user_id
                    )
                    if is_active:
                        start_new_chat()
                    st.rerun()

# ------------------------ Main Chat UI -------------------------------------#
st.markdown(
    "<h2 style='text-align: center; margin-bottom: 2rem;'>🤖 How can I help you today?</h2>",
    unsafe_allow_html=True,
)

current_messages = get_thread_state(st.session_state.thread_id)

for msg in current_messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg.content)
    elif isinstance(msg, ToolMessage):
        with st.expander(f"🔧 Tool Result: {msg.name}"):
            st.code(msg.content)


# ---------------------------------------------------------------------------#
# 2. Check for Paused State (Time-Delay HITL)
# ---------------------------------------------------------------------------#
config = {
    "configurable": {"thread_id": st.session_state.thread_id},
    "metadata": {"thread_id": st.session_state.thread_id},
}

graph_state = chatbot.get_state(config)  # type: ignore
is_paused = len(graph_state.next) > 0 and graph_state.next[0] == "tools"

if is_paused:
    last_message = graph_state.values["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    # Define which tools require a delay warning
    slow_tools = [
        "get_weather_data",
        "get_stock_price",
        "duckduckgo_search",
        "rag_tool",
    ]

    # Check if ANY of the AI's requested tools are in the slow list
    needs_approval = any(tc["name"] in slow_tools for tc in tool_calls)

    if not needs_approval:
        # SILENT AUTO-APPROVE: The tool is fast (e.g., calculator), instantly resume
        st.session_state.stream_input = None

    else:
        # MANUAL APPROVAL: The tool requires fetching external data
        tool_names = [tc["name"] for tc in tool_calls if tc["name"] in slow_tools]

        st.warning(f"⏳ The assistant needs to use **{', '.join(tool_names)}**.")
        st.info(
            "This operation connects to external services or searches large documents and may take a few moments. Do you wish to proceed?"
        )

        col1, col2, col3 = st.columns([2, 2, 8])
        with col1:
            if st.button("✅ Approve", use_container_width=True):
                st.session_state.stream_input = None
                st.rerun()
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                # If canceled, inject a mock ToolMessage back to the LLM
                # telling it the user aborted the action.
                rejection_messages = []
                for tc in tool_calls:
                    rejection_messages.append(
                        ToolMessage(
                            content="Action cancelled by the user. State that you cannot provide the information without running the tool.",
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        )
                    )
                chatbot.update_state(config, {"messages": rejection_messages}, as_node="tools")  # type: ignore
                st.session_state.stream_input = None
                st.rerun()

# ---------------------------------------------------------------------------#
# 3. Handle Normal Chat Input OR Resumed Stream
# ---------------------------------------------------------------------------#
elif user_input := st.chat_input("Type your message here..."):
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    # Store the input in session state to trigger the stream block
    st.session_state.stream_input = {"messages": [HumanMessage(content=user_input)]}


# If we have a new message OR we just clicked approve/cancel, run the stream
if "stream_input" in st.session_state:

    with st.chat_message("assistant", avatar="🤖"):
        status_container = st.container()
        response_placeholder = st.empty()

        assistant_response = ""
        tool_status = None

        try:
            # Stream using either the user input OR 'None' if resuming
            for event_chunk, metadata in chatbot.stream(
                st.session_state.stream_input,  # type: ignore
                config=config,  # type: ignore
                stream_mode="messages",
            ):
                if isinstance(event_chunk, AIMessageChunk) and getattr(
                    event_chunk, "tool_call_chunks", None
                ):
                    announced_ids = set()
                    for tc in event_chunk.tool_call_chunks:
                        if tc.get("name") and tc.get("id") not in announced_ids:
                            announced_ids.add(tc["id"])
                            if tool_status is None:
                                with status_container:
                                    tool_status = st.status(
                                        f"⚙️ Running `{tc['name']}`...", expanded=True
                                    )
                            else:
                                tool_status.write(f"⚙️ Running `{tc['name']}`...")

                elif isinstance(event_chunk, ToolMessage):
                    if tool_status is None:
                        with status_container:
                            tool_status = st.status(
                                "⚙️ Processing results...", expanded=True
                            )
                    tool_status.write(f"✅ Executed `{event_chunk.name}`")

                elif isinstance(event_chunk, AIMessageChunk) and event_chunk.content:
                    if tool_status:
                        tool_status.update(
                            label="✅ Tools executed successfully",
                            state="complete",
                            expanded=False,
                        )
                        tool_status = None

                    assistant_response += event_chunk.content  # type: ignore
                    response_placeholder.markdown(assistant_response + "▌")

            if assistant_response:
                response_placeholder.markdown(assistant_response)

        except Exception as exc:
            if tool_status:
                tool_status.update(
                    label="❌ Error executing tools", state="error", expanded=True
                )
            st.error(f"❌ An error occurred: {str(exc)}")

    # Clean up the trigger and refresh the UI state
    del st.session_state.stream_input
    st.session_state.chat_threads = retrieve_all_threads(st.session_state.user_id)
    st.rerun()
