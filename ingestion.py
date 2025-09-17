import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

from logger import log_info, log_success, log_header

load_dotenv()

if __name__ == "__main__":
    log_header("Starting ingestion")
    sunbird_dc_track = os.path.join("data", "dcTrack-API-Guide-7.0.0.pdf")
    loader = PyPDFLoader(sunbird_dc_track)
    documents = loader.load()
    log_info("Splitting text into chunks ...")
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    text_chunks = text_splitter.split_documents(documents)
    log_info(f"Created {len(text_chunks)} chunks")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=os.environ.get("INDEX_NAME_LANGSMITH"),
    )

    log_success("Finished ingestion")
