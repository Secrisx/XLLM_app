# X-LLM: Explainable Large Language Models

A tool for understanding and exploring the internal mechanisms of Large Language Models (LLMs) through multiple explanation methods.

## Project Overview

X-LLM provides comprehensive explanations of how different language models (GPT-2, OPT, Pythia) process and understand information. By analyzing both transformer attention patterns and feed-forward network activations, this tool enables researchers and enthusiasts to gain insights into these complex systems.

## Features

- **Multiple Model Support**: Currently supports GPT-2, OPT, and Pythia models
- **Dual Explanation Methods**:
  - **Transformer-based Explanations**: Analyze how attention heads focus on word relationships
  - **FFN-based Explanations**: Understand feed-forward network activations and word importance
- **Interactive UI**: Built with Streamlit for easy exploration
- **Retrieval-Augmented Generation (RAG)**: Uses semantic search to find relevant explanations based on user queries
- **Conversational Interface**: Chat-based interaction with the system

## Technical Architecture

- **Database**: PostgreSQL for storing model explanations
- **Embedding Model**: SentenceTransformer for semantic similarity search
- **LLM Agent**: GPT-4o-mini for generating human-friendly responses
- **Web Framework**: Streamlit for the user interface

## Setup Instructions

### Prerequisites

- Python 3.11+
- PostgreSQL server
- OpenAI API key

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Secrisx/XLLM_app.git
   cd XLLM_app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

