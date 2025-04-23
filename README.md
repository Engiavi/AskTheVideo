# 🎥 AskTheVideo – Smart Video Query Assistant Totally Based On YouTube

Skip the scrolling and stop wasting time watching full-length videos just to find the part you need. **AskTheVideo** lets you query a YouTube video with natural language and get intelligent, relevant answers based on the video’s content.

---

## 🚀 Features

- 🔗 Input any YouTube video URL
- 💬 Ask natural language questions about the video
- 🧠 Uses LLM (Large Language Model) to answer intelligently
- 📝 Parses transcripts using YouTube API
- 🧩 Efficient semantic search powered by embeddings & vector stores
- ⚡ Built using LangChain for streamlined, modular execution

---

## 🛠️ How It Works

1. **Transcript Extraction**
   - Fetch the transcript of a YouTube video using the YouTube API.

2. **Text Splitting & Vectorization**
   - Break the transcript into manageable chunks.
   - Generate vector embeddings for each chunk.
   - Store these embeddings in a vector store (e.g., FAISS, Pinecone).

3. **Retriever Setup**
   - When a user asks a question, use semantic search to find the most relevant chunks from the vector store.

4. **Prompt Construction**
   - Form a rich prompt using the selected transcript chunks and the user's query.

5. **LLM Response**
   - Send the prompt to an LLM (e.g., OpenAI GPT-4, Claude, etc.) to generate a detailed, context-aware response.

6. **LangChain Integration**
   - Use LangChain’s chain framework to handle the end-to-end process: from transcript retrieval to final answer.

---

## 🧰 Tech Stack

- **Python**
- **LangChain**
- **OpenAI / Other LLMs**
- **FAISS / Pinecone / Chroma (Vector DB)**
- **YouTube Data API v3**

---

## 📦 Installation

```bash
git clone https://github.com/Engiavi/AskTheVideo.git
cd AskTheVideo
python -m venv venv
venv/Scripts/Activate
pip install -r requirements.txt
