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

# Set page configuration and title
st.set_page_config(
    page_title="X-LLM: Multi-model Explanations",
    page_icon="🧠",
    layout="wide"
)

# Initialize connection to database
@st.cache_resource
def init_db():
    return psycopg2.connect(dbname="xllm", user="postgres", host="localhost", port=5432)

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
         for model_name, layer, head, words, _ in results_transformer[:5]]
    )
    
    retrieved_context += "\n\nFFN Data:\n" + "\n".join(
        [f"Model: {model_name.upper()}, Layer: {layer}, Score: {score}, Word: {word[:1000]}" 
         for model_name, layer, score, word, _ in results_ffn[:5]]
    )

    return retrieved_context

def generate_response(user_query, llm, cursor, embedding_model, selected_models, chat_history):
    # Model keywords for detection
    model_keywords = {
        "gpt2": ["gpt2", "gpt-2", "gpt 2", "gpt2 model"],
        "opt": ["opt", "opt model", "opt-model", "the opt", "facebook opt", "meta opt"],
        "pythia": ["pythia", "pythia model"]
    }

    # Check if user is trying to select specific models
    if user_query.lower().startswith("use model"):
        model_names = [m.strip().lower() for m in user_query[10:].split(",")]
        valid_models = [name for name in MODELS.keys() if any(keyword in model_names for keyword in model_keywords[name])]
        
        if valid_models:
            return f"Now using models: {', '.join([name.upper() for name in valid_models])}", valid_models
        else:
            return f"Invalid model selection. Available models: {', '.join(MODELS.keys())}", selected_models
    
    # Check if specific models are mentioned in the query
    mentioned_models = [model for model in MODELS.keys() 
                        if any(keyword in user_query.lower() for keyword in model_keywords[model])]
    
    # Only change the model selection if models are mentioned
    if mentioned_models:
        selected_models = mentioned_models
    
    # Retrieve context based on the query and selected models
    context = retrieved_context(user_query, cursor, embedding_model, selected_models)
    
    system_prompt = f"""
        You are an assistant which answers questions based on knowledge which is provided to you.
        The database contains explanations from 3 large language models: GPT-2, OPT, and Pythia.

        There are two types of explanations:
        1) Transformer-based explanations with structure [model, layer, head, word_pairs], which means 
           that the specific attention head in the layer is particularly attuned to the relationships 
           of the word pairs in the lists
        
        2) FFN-based explanations with structure [model, layer, score, word] which corresponds to the 
           feed-forward network of each layer, with the words listing the most relevant terms captured 
           by the FFN and the corresponding scores showing how strongly they align with the concept 
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
    
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
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

# Main application
def main():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I can help you understand how different language models (GPT-2, OPT, and Pythia) work internally. What would you like to know about their transformer or feed-forward network mechanisms?"}
        ]
    
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = list(MODELS.keys())
    
    # Display title and description
    st.title("🧠 X-LLM: Understanding Language Model Explanations")
    st.markdown("""
    This application provides insights into how different language models work internally through two explanation methods:
    * **Transformer-based explanations**: Shows which attention heads are attuned to specific word relationships
    * **FFN-based explanations**: Reveals the words and concepts captured by feed-forward networks
    
    Available models: GPT-2, OPT, Pythia
    
    You can type "use model [model1], [model2]" to focus on specific models.
    """)
    
    # Display current model selection
    st.sidebar.header("Selected Models")
    st.sidebar.write(", ".join(model.upper() for model in st.session_state.selected_models))
    
    # Model selection via sidebar
    st.sidebar.header("Choose Models")
    model_selection = {}
    for model in MODELS.keys():
        model_selection[model] = st.sidebar.checkbox(model.upper(), 
                                                  value=model in st.session_state.selected_models)
    
    # Update selected models if changed via checkboxes
    selected_via_checkbox = [model for model, selected in model_selection.items() if selected]
    if selected_via_checkbox and selected_via_checkbox != st.session_state.selected_models:
        st.session_state.selected_models = selected_via_checkbox
    
    # Initialize resources
    try:
        db = init_db()
        cursor = db.cursor()
        embedding_model = load_embedding_model()
        llm = load_llm()
    except Exception as e:
        st.error(f"Failed to initialize resources: {str(e)}")
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about language model internal mechanisms..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, updated_models = generate_response(
                    prompt, 
                    llm, 
                    cursor, 
                    embedding_model, 
                    st.session_state.selected_models,
                    st.session_state.messages[:-1]  # Exclude the just-added user message
                )
                st.markdown(response)
                
        # Update selected models if changed by the response
        st.session_state.selected_models = updated_models
                
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()