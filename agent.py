
import os

from dotenv import load_dotenv
from langchain.agents import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pinecone import Pinecone

load_dotenv()

class SunbirdAgent:
    _instance = None

    def __init__(self):
        if SunbirdAgent._instance is not None:
            raise Exception("This class is a singleton!")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.pinecone = Pinecone()
        self.vector_store = self.pinecone.Index(os.getenv("INDEX_NAME_LANGSMITH"))

        @tool
        def sunbird_dc_track_api_doc(query: str):
            """Searches the Sunbird DC Track API documentation for relevant context."""
            vector = self.embeddings.embed_query(query)
            return self.vector_store.query(vector=vector, top_k=10, include_metadata=True)

        self.tools = [sunbird_dc_track_api_doc]
        self.agent = create_react_agent(
            tools=self.tools,
            model=ChatOpenAI(model="gpt-4o", temperature=0.0).bind_tools(self.tools)
        )
        SunbirdAgent._instance = self

    @staticmethod
    def get_instance():
        if SunbirdAgent._instance is None:
            SunbirdAgent()
        return SunbirdAgent._instance.agent

sunbird_agent = SunbirdAgent.get_instance()