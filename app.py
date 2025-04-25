import streamlit as st
import psycopg2
import os
from sentence_transformers import SentenceTransformer
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv

#Load environment variables
load_dotenv()

# Define available models
MODELS = {
    "gpt2": 1,
    "opt": 2,
    "pythia": 3
}

# Define preset questions
PRESET_QUESTIONS = [
    "How does GPT-2 understand geographical concepts?",
    "Compare how different models process emotional words",
    "What attention patterns exist in OPT for handling numerical information?",
    "How does Pythia represent abstract concepts in its FFN layers?",
    "Show differences in how models understand language syntax"
]

# Set page configuration and title
st.set_page_config(
    page_title="X-LLM: Multi-model Explanations",
    page_icon="🧠",
    layout="wide"
)

# Initialize connection to database
@st.cache_resource
def init_db():
    return psycopg2.connect(dbname="xllm", user="postgres", host="lovelace.deac.wfu.edu", port=5432)

# Initialize embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Initialize LLM
@st.cache_resource
def load_llm():
    # Set OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = openai_key
    return init_chat_model("gpt-4o-mini", model_provider="openai")

# Functions to retrieve context from the database
def retrieved_context(user_query, cursor, embedding_model, selected_models=None):
    query_emb = embedding_model.encode([user_query])[0]
    
    if not selected_models:
        selected_models = list(MODELS.keys())

    results_transformer = []
    results_ffn = []
    
    for model in selected_models:
        try:
            cursor.execute(f"""
                SELECT layer, head, word_pair, embedding <=> %s::vector AS similarity
                FROM {model}_transformer
                ORDER BY similarity ASC
                LIMIT 5;
            """, (query_emb.tolist(),))
    
            model_results = cursor.fetchall()

            for result in model_results:
                results_transformer.append((model,) + result)
        except Exception as e:
            st.error(f"Error querying {model}_transformer: {str(e)}")
    
        try:
            cursor.execute(f"""
                SELECT layer, score, word, embedding <=> %s::vector AS similarity
                FROM {model}_ffn
                ORDER BY similarity ASC
                LIMIT 5;
            """, (query_emb.tolist(),))
    
            model_results = cursor.fetchall()
            
            for result in model_results:
                results_ffn.append((model,) + result)
        except Exception as e:
            st.error(f"Error querying {model}_ffn: {str(e)}")
           
    retrieved_context = "Transformer Data:\n" + "\n".join(
        [f"Model: {model_name.upper()}, Layer: {layer}, Head: {head}, Words: {words[:1000]}" 
         for model_name, layer, head, words, _ in results_transformer[:7]]
    )
    
    retrieved_context += "\n\nFFN Data:\n" + "\n".join(
        [f"Model: {model_name.upper()}, Layer: {layer}, Score: {score}, Word: {word[:1000]}" 
         for model_name, layer, score, word, _ in results_ffn[:7]]
    )

    return retrieved_context

def generate_response(user_query, llm, cursor, embedding_model, selected_models, chat_history):
    model_keywords = {
        "gpt2": ["gpt2", "gpt-2", "gpt 2", "gpt2 model"],
        "opt": ["opt", "opt model", "opt-model", "the opt", "facebook opt", "meta opt"],
        "pythia": ["pythia", "pythia model"]
    }
    
    try:
        mentioned_models = [model for model in MODELS.keys() 
                        if any(keyword in user_query.lower() for keyword in model_keywords[model])]
        
        # Only change the model selection if models are mentioned
        if mentioned_models:
            selected_models = mentioned_models
        
        # Safety check - if we end up with no models, use all models
        if not selected_models:
            selected_models = list(MODELS.keys())
    except Exception as e:
        print(f"Error in model detection: {e}")
        # Fall back to all models if there's an error
        selected_models = list(MODELS.keys())
    
    # Retrieve context based on the query and selected models
    context = retrieved_context(user_query, cursor, embedding_model, selected_models)
    
    if context:
        context = context.replace("{", "{{").replace("}", "}}")
    
    system_prompt = f"""
        You are an assistant which answers questions based on knowledge which is provided to you.
        The database contains explanations from 3 large language models: GPT-2, OPT, and Pythia.

        There are two types of explanations:
        1) Transformer-based explanations with structure [model, layer, head, word_pairs], which means 
           that in the model, the specific attention head in the layer is particularly attuned to the 
           relationships of the word pairs in the lists
        
        2) FFN-based explanations with structure [model, layer, score, word] which corresponds to the 
           feed-forward network of each layer in the model, with the words listing the most relevant terms 
           captured by the FFN and the corresponding scores showing how strongly they align with the concept 
           encoded in that layer.
        
        While answering, you should mainly focus on the information in the "Context" section other 
        than your own knowledge. You don't need to mention anything to the user about the provided knowledge.

        When asked by users and appropriate, compare and contrast how different models (GPT-2, OPT, Pythia) understand 
        or process similar concepts. Highlight any interesting similarities or differences between models.

        Maintain a conversational tone and consider the conversation history when answering,
        but don't need to talk anything about "feel free to ask more questions". If the user 
        refers to previous questions or information, understand these references in the context 
        of the ongoing conversation.

        Context:
        {context or "No relevant context found."}
        """
    
    from langchain_core.prompts.chat import  ChatPromptTemplate, MessagesPlaceholder
    
    # Create a message history from the chat_history
    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    
    # Add the current user query
    messages.append(HumanMessage(content=user_query))
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    prompt = prompt_template.invoke({"messages": messages})
    response = llm.invoke(prompt)
    
    return response.content, selected_models

# Function to handle shortcut button clicks
def process_shortcut(question):
    # Add the question to the session state
    st.session_state.question_to_ask = question


# Add custom CSS for centering content
def main():
    # Add custom CSS for centering the chat container
    st.markdown("""
    <style>
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I can help you understand how different language models work internally. What would you like to know about their mechanisms?"}
        ]
    
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = list(MODELS.keys())
        
    if "question_to_ask" not in st.session_state:
        st.session_state.question_to_ask = None
    
    # Process any question from shortcut button
    if st.session_state.question_to_ask:
        question = st.session_state.question_to_ask
        st.session_state.question_to_ask = None  # Reset after processing
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
    
    # Display title and description
    st.title("🧠XLLM: Understanding Large Language Model Explanations")
    st.markdown("""
    This application provides insights into how different language models work internally through two explanation methods:
    * **Transformer-based explanations**: Shows which attention heads are attuned to specific word relationships
    * **FFN-based explanations**: Reveals the words and concepts captured by feed-forward networks
    
    Current available models: GPT-2, OPT, Pythia
        
    Enjoy your explorations to the models!
    """)
    
    # Initialize resources
    try:
        db = init_db()
        cursor = db.cursor()
        embedding_model = load_embedding_model()
        llm = load_llm()
    except Exception as e:
        st.error(f"Failed to initialize resources: {str(e)}")
        return
    
    # Create a centered container for the chat
    left_col, center_col, right_col = st.columns([1, 2, 1])
    
    with center_col:
        # Display chat history in centered column
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process the last message if it's from the user and hasn't been responded to
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            user_query = st.session_state.messages[-1]["content"]
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response, updated_models = generate_response(
                            user_query, 
                            llm, 
                            cursor, 
                            embedding_model, 
                            st.session_state.selected_models,
                            st.session_state.messages[:-1]
                        )
                        st.markdown(response)
                        
                        # Update selected models if changed by the response
                        st.session_state.selected_models = updated_models
                            
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error while processing your question."})
        
        # Chat input in the centered column
        if prompt := st.chat_input("Ask about model internal mechanisms..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()  # Rerun to show the user message and generate response
    
    # Add a divider
    st.markdown("---")
    
    # Add shortcut buttons section below the chat input
    st.subheader("Try these example questions:")
    
    # Create columns for buttons to arrange them in a row
    # We're using 5 columns - empty space on sides for better alignment
    col1, col2, col3, col4, col5 = st.columns([0.5, 1, 1, 1, 0.5])
    
    # Add first three buttons in first row
    with col2:
        st.button("🌍 Geography Concepts", 
                 key="geo_btn", 
                 help="Ask about how models understand geography",
                 on_click=process_shortcut, 
                 args=(PRESET_QUESTIONS[0],),
                 use_container_width=True)
    
    with col3:
        st.button("😊 Emotional Processing", 
                 key="emotion_btn", 
                 help="Compare how models process emotions",
                 on_click=process_shortcut, 
                 args=(PRESET_QUESTIONS[1],),
                 use_container_width=True)
    
    with col4:
        st.button("🔢 Numerical Understanding", 
                 key="num_btn", 
                 help="Explore numerical processing in models",
                 on_click=process_shortcut, 
                 args=(PRESET_QUESTIONS[2],),
                 use_container_width=True)
    
    # Create second row of buttons
    col1, col2, col3, col4, col5 = st.columns([0.5, 1, 1, 1, 0.5])
    
    with col2:
        st.button("🧠 Abstract Concepts", 
                 key="abstract_btn", 
                 help="Explore how models handle abstract ideas",
                 on_click=process_shortcut, 
                 args=(PRESET_QUESTIONS[3],),
                 use_container_width=True)
    
    with col3:
        st.button("📝 Language Syntax", 
                 key="syntax_btn", 
                 help="Compare syntax understanding across models",
                 on_click=process_shortcut, 
                 args=(PRESET_QUESTIONS[4],),
                 use_container_width=True)

if __name__ == "__main__":
    main()