from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv() # load environment variables from .env file

model = ChatGoogleGenerativeAI(model= "google-gemini-1.5-turbo")

# step 1 :- Indexing (Document Ingesetion)
video_id = "J5_-l7WIO_w" # dealing with video id instead of url, in future we can add a function to extract video id from url

