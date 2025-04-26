import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env file

model = ChatGoogleGenerativeAI(model= "google-gemini-1.5-turbo")

# Step 1: Indexing (Document Ingestion)
video_id = "Gfr50f6ZBvo"
vector_store_path = f"vector_store_{video_id}"

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

if os.path.exists(vector_store_path):
    print("Loading existing vector store...")
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)

else:
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        # print(transcript)
    except TranscriptsDisabled:
        print("No captions available for this video.")
        transcript = ""

    # step 1b :- Indexing (Text Splitter)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    
    # print(len(chunks)) #
    # Step 1c & 1d: Embedding Generation and Storing in Vector Store
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(vector_store_path)
    print("Vector store created and saved.")

# Print all document IDs
# print(vector_store.index_to_docstore_id)

# To see a particular document by ID (example usage)
# print(vector_store.docstore._dict['d634eaab-9733-45d3-8cd4-9d9da8737873'])


#step2 : Retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
# print(retriever) #it gives the information about the retriever, i.e. which vector stores and embedding models  
print(retriever.invoke('What is deepmind'))