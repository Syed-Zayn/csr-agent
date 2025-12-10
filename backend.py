import os
import time  # New Import for waiting
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

# BRAIN: Google Gemini 1.5 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0, 
    streaming=True
)

# SEARCH TOOL: Gemini Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Connect to Pinecone
vector_store = PineconeVectorStore(
    index_name="csr-agent-gemini",
    embedding=embeddings
)

# K=10 for Deep Search
retriever = vector_store.as_retriever(search_kwargs={"k": 20})

# --- 2. Setup Neon DB (Memory) ---
connection_string = os.getenv("NEON_DB_URL")
if connection_string:
    db_url = connection_string.replace("+asyncpg", "").replace("+psycopg", "")
else:
    db_url = "" # Handle generic case if needed

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
    
    try:
        docs = retriever.invoke(latest_message)
        context_text = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"⚠️ Retrieval Error: {e}")
        context_text = "No context available due to error."
    
    return {"context": context_text}

def generate_node(state: AgentState):
    """
    Answer Step: Uses Gemini LLM with AUTO-RETRY for 429 Errors.
    """
    
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
    
    # --- RETRY LOGIC START ---
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Try to generate response
            response = chain.invoke({"context": state["context"], "messages": state["messages"]})
            return {"messages": [response]}
            
        except Exception as e:
            error_msg = str(e)
            # Check for Rate Limit (429) or Resource Exhausted
            if "429" in error_msg or "ResourceExhausted" in error_msg:
                wait_time = 12 * (attempt + 1) # Wait 12s, then 24s...
                print(f"⚠️ Google API Limit Hit (429). Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                # Agar koi aur error hai to crash hone do
                raise e
    
    # Agar 3 baar try karne ke baad bhi na chale
    return {"messages": [AIMessage(content="⚠️ System is currently busy (Google API Rate Limit). Please try again in a minute.")]}
    # --- RETRY LOGIC END ---

# --- 5. Graph Build ---
def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    if db_url:
        conn = Connection.connect(db_url)
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()
        return workflow.compile(checkpointer=checkpointer)
    else:
        return workflow.compile()

graph = build_graph()
