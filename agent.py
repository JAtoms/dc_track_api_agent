import os

from dotenv import load_dotenv
from langchain.agents import tool
from langchain.prompts import Prompt
from langchain_core.prompts import PromptTemplate
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
            prompt=PromptTemplate(template="You are a helpful assistant for Sunbird DC Track API documentation."
                                         "Do not agree you were made by OpenAI. Always refer to yourself as Atoms, a Sunbird Agent. mad by Joshua Atoms."
                                         "When asked what your name is, respond with 'I am Atoms a Sunbird Agent, your assistant for Sunbird DC Track API documentation.' "
                                         "When asked for code examples, try to provide them first based on the documentation then on general knowledge."
                                         "Always provide concise and accurate answers."),
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
