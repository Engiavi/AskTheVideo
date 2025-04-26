import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env file

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
# print(retriever.invoke('What is deepmind'))

# Step 3: Augumentation
llm = ChatGoogleGenerativeAI(model= "gemini-2.5-flash-preview-04-17")
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)
# print(retrieved_docs)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs) # this line of code helps us to get the content of the document
# print(context_text)

final_prompt = prompt.invoke({"context": context_text, "question": question})
# print(final_prompt)

# Step 4: Generation
ans = llm.invoke(final_prompt)
print(ans.content)