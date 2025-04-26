from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv() # load environment variables from .env file

model = ChatGoogleGenerativeAI(model= "google-gemini-1.5-turbo")

# step 1 :- Indexing (Document Ingesetion)
video_id = "Gfr50f6ZBvo" # dealing with video id instead of url, in future we can add a function to extract video id from url
try:
    # If you don’t care which language, this returns the “best” one
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
 
    # Flatten it to plain text
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")
    

# step 1b :- Indexing (Text Splitter)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

print(len(chunks))