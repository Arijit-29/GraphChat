# GraphChat

> A colorful, lightweight AI assistant built with Streamlit and LangGraph for chatting, document Q&A, and tool-based tasks.

GraphChat is a simple but powerful chat application that lets you talk to an AI assistant, upload PDFs for document-based answers, and use helpful tools for search, calculations, weather, and stock data.

## ✨ What the project does

- Start new chats and switch between conversation threads
- Upload a PDF and use it as document context for question answering
- Ask questions about the uploaded file and get answers based on retrieved content
- Use built-in tools for web search, calculations, weather, and stock lookup

## 🚀 Main features

- Thread-based chat history
- PDF ingestion and indexing for RAG
- Document-aware responses from uploaded content
- Simple approval flow for slower external tools
- Clean and friendly Streamlit interface

## 🛠️ Tool functionality

The assistant can use these tools:

- `rag_tool` — searches the uploaded PDF for relevant information
- `calculator` — performs basic arithmetic operations
- `get_stock_price` — fetches stock price information
- `get_weather_data` — retrieves weather forecast details
- `search_tool` — performs a web search

## ⚙️ Setup

1. Install the required Python packages:

   ```bash
   git clone https://github.com/Arijit-29/GraphChat.git
   cd GraphChat
   pip install -r requirements.txt
   ```

2. Create a `.env` file and add your API keys:

   ```env
   GROQ_API_KEY=your_groq_key
   OPENWEATHER_API_KEY=your_weather_api_key
   ALPHA_VANTAGE_API_KEY=your_stock_api_key
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

## 🧰 Tech stack

- Python
- Streamlit for the web interface
- LangGraph for the conversation workflow
- LangChain for model and tool integration
- Chroma for vector storage and retrieval
- Hugging Face embeddings for PDF content search
- Groq LLM for chat responses
