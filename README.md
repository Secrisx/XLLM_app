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
   git clone https://github.com/yourusername/X-LLM.git
   cd X-LLM
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Create PostgreSQL database:
   ```bash
   createdb xllm
   ```

### Database Schema

Ensure your PostgreSQL database has the following tables for each model:
- `[model]_transformer`: Stores transformer-based explanations
- `[model]_ffn`: Stores FFN-based explanations

Each table should include vector embeddings for semantic search.

### Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

## Adding New Models

To add a new model to the system:

1. Generate explanation data for the model
2. Create tables `[model_name]_transformer` and `[model_name]_ffn` in the database
3. Update the `MODELS` dictionary in `app.py`
4. Add appropriate keywords to the `model_keywords` dictionary in the `generate_response` function

## Contact

For questions and support, please open an issue on GitHub or contact [your-email@example.com].

## License

This project is licensed under the MIT License - see the LICENSE file for details.
