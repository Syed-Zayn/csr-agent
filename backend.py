import os
import time
import sqlite3  # ✅ Changed: Using SQLite (Local DB) instead of Postgres to fix SSL errors
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

# --- IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver # ✅ Changed: Imports SQLite Saver
load_dotenv()

# --- 1. Setup Models ---

# BRAIN: Google Gemini 1.5 Flash (Standard Free Model)
# ✅ FIX: Using 'gemini-1.5-flash' because it is the stable free version
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0, 
    streaming=True
)

# SEARCH TOOL: Gemini Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Connect to Pinecone (Vector Database)
vector_store = PineconeVectorStore(
    index_name="csr-agent-gemini",
    embedding=embeddings
)

# K=10 means it will search for top 10 relevant pages
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# --- 2. Setup Database (SQLite) ---
# We are using a local file 'checkpoints.db' for memory.
# This fixes the "SSL Connection Closed" error permanently.
DB_PATH = "checkpoints.db"

# --- 3. LangGraph State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The chat history"]
    context: str

# --- 4. Nodes Logic ---

def retrieve_node(state: AgentState):
    """
    Search Step: Uses Gemini Embeddings to find data in Pinecone.
    """
    # Get the last message from the user
    latest_message = state["messages"][-1].content
    print(f"🔍 Searching Pinecone for: {latest_message}")
    
    try:
        # Perform retrieval
        docs = retriever.invoke(latest_message)
        # Combine found documents into a single string
        context_text = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"⚠️ Retrieval Error: {e}")
        context_text = "No context available due to error."
    
    return {"context": context_text}

def generate_node(state: AgentState):
    """
    Answer Step: Uses Gemini LLM with AUTO-RETRY logic for API limits.
    """
    
    # ✅ FIX: REMOVED 'CSR i Praktiken' completely.
    # Replaced with 'Sustainability in Practice' in the persona definition.
    system_prompt = (
        "You are an expert Corporate Social Responsibility (CSR) Consultant. "
        "Your role is to advise clients strictly based on the provided case studies and articles. "
        "You represent the knowledge contained in 'Sustainability in Practice'.\n\n"
        
        "### CRITICAL INSTRUCTIONS:\n"
        "1. **STRICT CONTEXT USE:** Answer ONLY using the provided Context. Do not use outside knowledge.\n"
        "2. **SOURCE NAME:** Always refer to the source as **'Sustainability in Practice'**. NEVER use the Swedish name.\n"
        "3. **MISSING INFO:** If the exact answer isn't in the context, say: 'I checked the case studies, but this specific detail is not mentioned.'\n"
        "4. **GREEN CONSULTANT:** If asked about 'Green Consultant', refer to the section about advising on eco-friendly building (Lauren Gropper).\n"
        "5. **TONE:** Professional, evidence-based, and helpful.\n"
        "6. **LANGUAGE:** Answer in the same language as the user (English or Portuguese).\n"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("system", "CONTEXT FROM DATABASE:\n{context}"),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    chain = prompt | llm
    
    # --- FREE TIER SAFETY LOGIC (Delay + Retry) ---
    
    # Initial sleep to prevent hitting rate limits too fast
    time.sleep(2)

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
                # If it's a different error, raise it
                raise e
    
    # If all 3 attempts fail
    return {"messages": [AIMessage(content="⚠️ System is currently busy (Google API Rate Limit). Please try again in a minute.")]}
    # --- RETRY LOGIC END ---

# --- 5. Build Graph (SQLite Version) ---
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    try:
        # ✅ FIX: Using SQLite Connection (Local File)
        # check_same_thread=False is needed for Streamlit
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        
        # Setup the Checkpointer (Memory)
        checkpointer = SqliteSaver(conn)
        
        # Compile the graph with memory
        return workflow.compile(checkpointer=checkpointer)
        
    except Exception as e:
        print(f"⚠️ Memory Error: {e}")
        # Fallback: Run without memory if database fails (Very unlikely with SQLite)
        return workflow.compile()

# Initialize the graph
graph = build_graph()
