import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

# Sirf Google ki Libraries use hongi
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection

load_dotenv()

# --- 1. Setup Models ---

# BRAIN: Google Gemini 2.5 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0, # Strict factual mode
    streaming=True
)

# SEARCH TOOL: Gemini Embeddings (Must match ingest.py)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Connect to Pinecone
vector_store = PineconeVectorStore(
    index_name="csr-agent-gemini", # Ensure same name as ingest.py
    embedding=embeddings
)

# K=10 for Deep Search
retriever = vector_store.as_retriever(search_kwargs={"k": 40})

# --- 2. Setup Neon DB (Memory) ---
connection_string = os.getenv("NEON_DB_URL")
db_url = connection_string.replace("+asyncpg", "").replace("+psycopg", "")

# --- 3. LangGraph State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The chat history"]
    context: str

# --- 4. Nodes Logic ---

def retrieve_node(state: AgentState):
    """
    Search Step: Uses Gemini Embeddings to find data in Pinecone.
    """
    latest_message = state["messages"][-1].content
    print(f"🔍 Searching Pinecone for: {latest_message}")
    
    docs = retriever.invoke(latest_message)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    return {"context": context_text}

def generate_node(state: AgentState):
    """
    Answer Step: Uses Gemini LLM to write the answer based on Context.
    """
    
    # SYSTEM PROMPT (Optimized for your client's needs)
    system_prompt = (
        "You are an expert Corporate Social Responsibility (CSR) Consultant. "
        "Your role is to advise clients strictly based on the provided case studies and articles. "
        "You represent the knowledge contained in 'CSR i Praktiken'.\n\n"
        
        "### CRITICAL INSTRUCTIONS:\n"
        "1. **STRICT CONTEXT USE:** Answer ONLY using the provided Context. Do not use outside knowledge.\n"
        "2. **MISSING INFO:** If the exact answer isn't in the context, say: 'I checked the case studies, but this specific detail is not mentioned.'\n"
        "3. **GREEN CONSULTANT:** If asked about 'Green Consultant', refer to the section about advising on eco-friendly building (Lauren Gropper).\n"
        "4. **TONE:** Professional, evidence-based, and helpful.\n"
        "5. **LANGUAGE:** Answer in the same language as the user (English or Portuguese).\n"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("system", "CONTEXT FROM DATABASE:\n{context}"),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({"context": state["context"], "messages": state["messages"]})
    
    return {"messages": [response]}

# --- 5. Graph Build ---
def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    conn = Connection.connect(db_url)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    
    return workflow.compile(checkpointer=checkpointer)

graph = build_graph()