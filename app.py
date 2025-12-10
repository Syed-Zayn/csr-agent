import streamlit as st
import uuid
import time
from langchain_core.messages import HumanMessage, AIMessage
from backend import graph  # Ensure backend.py is connected properly

# --- 1. Page Config ---
st.set_page_config(
    page_title="Eco-CSR Consultant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS (Modern & Clean) ---
st.markdown("""
<style>
    /* Force Text Color to Black */
    .stApp, .stMarkdown, p, h1, h2, h3 {
        color: #31333F !important;
    }

    /* Input Box Styling (Fix Invisible Text) */
    .stTextInput input {
        color: #31333F !important;
        background-color: #ffffff !important;
        border: 1px solid #4CAF50;
    }
    
    /* Header Styling */
    h1 {
        color: #2E7D32 !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #ddd;
    }
    
    /* Chat Message Bubbles */
    .stChatMessage {
        background-color: transparent;
        border: none;
    }
    
    /* User Message Bubble */
    div[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #e8f5e9; /* Light Green */
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #c8e6c9;
    }

    /* AI Message Bubble */
    div[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Button Styling */
    div.stButton > button {
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Session State Logic ---

# Initialize Chat Sessions
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

# Ensure Current Thread Exists
if "current_thread_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.current_thread_id = new_id
    st.session_state.chat_sessions[new_id] = {'title': 'New Conversation', 'messages': []}

def get_current_messages():
    return st.session_state.chat_sessions[st.session_state.current_thread_id]['messages']

# --- 4. Sidebar Interface ---
with st.sidebar:
    # Logo Area
    st.markdown("<div style='text-align: center; margin-bottom: 20px;'>", unsafe_allow_html=True)
    # Using a professional Icon URL
    st.image("https://cdn-icons-png.flaticon.com/512/2913/2913465.png", width=80)
    st.markdown("<h2 style='color: #1b5e20;'>CSR Agent</h2></div>", unsafe_allow_html=True)
    
    # New Chat Button
    if st.button("➕ New Chat", type="primary", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.current_thread_id = new_id
        st.session_state.chat_sessions[new_id] = {'title': 'New Conversation', 'messages': []}
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📂 Chat History")
    
    # History List
    session_ids = list(st.session_state.chat_sessions.keys())
    for s_id in reversed(session_ids):
        session = st.session_state.chat_sessions[s_id]
        label = session['title']
        if len(label) > 22: label = label[:22] + "..."
        
        # Style active chat differently
        if s_id == st.session_state.current_thread_id:
            st.button(f"🟢 {label}", key=s_id, disabled=True, use_container_width=True)
        else:
            if st.button(f"💬 {label}", key=s_id, use_container_width=True):
                st.session_state.current_thread_id = s_id
                st.rerun()

# --- 5. Main Chat Interface ---

# Header
st.markdown("""
<div style='text-align: center; margin-bottom: 40px;'>
    <h1>🌿 Corporate Social Responsibility Expert</h1>
    <p style='color: #666; font-size: 1.1em;'>
        Ask me about <b>Green Consultants, IKEA, Starbucks</b>, and more from 'CSR i Praktiken'.
    </p>
</div>
""", unsafe_allow_html=True)

# Display Messages
current_messages = get_current_messages()

for msg in current_messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg.content)

# --- 6. Chat Logic ---
if user_input := st.chat_input("Type your question here..."):
    
    # Add User Message
    current_messages.append(HumanMessage(content=user_input))
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Update Title (First message becomes title)
    if len(current_messages) == 1:
        st.session_state.chat_sessions[st.session_state.current_thread_id]['title'] = user_input

    # AI Response
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        
        config = {"configurable": {"thread_id": st.session_state.current_thread_id}}
        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        try:
            # Graph Streaming
            for event in graph.stream(inputs, config=config, stream_mode="values"):
                if "messages" in event:
                    last_msg = event["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        full_response = last_msg.content
                        # Simulate typing speed slightly for better feel
                        message_placeholder.markdown(full_response + "▌")
            
            # Final text
            message_placeholder.markdown(full_response)
            current_messages.append(AIMessage(content=full_response))
            
        except Exception as e:
            st.error(f"⚠️ Error: {e}")