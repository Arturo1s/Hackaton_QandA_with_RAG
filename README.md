# PLV Intelligent Help Center - RAG + LLaMA

A Retrieval-Augmented Generation (RAG) application that provides intelligent answers to student questions about PLV (Pôle Léonard de Vinci) schools using FAISS vector search and the Llama 3.1 language model.

## Features

- **Semantic Search**: Uses sentence transformers to encode documents and FAISS for fast vector similarity search
- **RAG Architecture**: Combines retrieved context from a knowledge base with LLM generation for accurate answers
- **Multi-School Support**: Handles questions for EMLV, ESILV, and EXECUTIVE Education schools
- **Interaction Logging**: Tracks all user queries and responses for analysis
- **Streamlit UI**: Clean, user-friendly web interface
- **Error Recovery**: Automatic index rebuilding on corruption

## Project Structure

```
.
├── app.py                  # Streamlit application interface
├── rag_engine.py          # RAG engine with FAISS indexing
├── llm_client.py          # LLM API client (HuggingFace Inference)
├── requirements.txt       # Python dependencies
├── data/
│   └── questions.csv      # Knowledge base (questions and answers)
├── vectorstore/
│   ├── faiss_index.bin    # FAISS index file
│   └── embeddings.npy     # Sentence embeddings
├── logs/
│   └── interactions.csv   # User interaction history
└── __pycache__/           # Python cache
```

## Installation

### Prerequisites
- Python 3.8+
- HuggingFace API key for LLM access

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Hackaton_QandA_with_RAG
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root:
```
HF_API_KEY=your_huggingface_api_key_here
```

## Usage

### Start the Application

Run the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### How It Works

1. **User Input**: Enter your question in French in the text input field
2. **Search**: The system searches the knowledge base using semantic similarity
3. **Generate**: Retrieved context is passed to Llama 3.1 for answer generation
4. **Display**: The answer is formatted in HTML and displayed with sources
5. **Log**: The interaction is automatically logged for analytics

## Configuration

### Data Format

The `data/questions.csv` file uses semicolon (`;`) delimiters and contains:
- `id`: Question ID
- `Title`: Question title
- `Content`: Detailed answer
- `Date`: Publication date
- `Post Type`: Type of content
- `Langues`: Language
- `Thématiques`: Topic categories
- `Utilisateurs`: Target users (student, staff, faculty)
- `Écoles`: Associated schools
- `Status`: Publication status

### Model Configuration

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **LLM Model**: `meta-llama/Llama-3.1-8B-Instruct`
- **Provider**: HuggingFace Inference API
- **Vector Database**: FAISS IndexFlatL2

## Components

### `rag_engine.py`
Handles the RAG pipeline:
- `load_dataset()`: Reads CSV data with proper encoding and delimiter handling
- `build_faiss_index()`: Creates embeddings and builds FAISS index
- `load_faiss()`: Loads or rebuilds the index with error recovery
- `search()`: Performs semantic search on user queries

### `llm_client.py`
Manages LLM interactions:
- Initializes HuggingFace Inference Client
- `generate_answer()`: Sends prompts to Llama 3.1 via chat completion API

### `app.py`
Streamlit application:
- User interface for queries
- RAG pipeline orchestration
- Logging of interactions
- Display of sources and answers

## Error Handling

The application includes robust error handling:
- **Corrupted Index**: Automatically detects and rebuilds FAISS index
- **Empty Log Files**: Recovers from corrupted interaction logs
- **Encoding Issues**: Handles `latin-1` encoding in CSV files
- **CSV Parsing**: Skips malformed rows with embedded line breaks

## Performance

- **Embedding Generation**: ~384-dim vectors using efficient sentence-transformers
- **Search Speed**: Instant with FAISS IndexFlatL2
- **LLM Response**: ~5-10 seconds depending on API latency
- **Scalability**: Supports thousands of Q&A pairs

## Logging

User interactions are logged to `logs/interactions.csv` with:
- `timestamp`: When the query was made
- `question`: User's question
- `answer`: Generated answer
- `source_found`: Whether a relevant source was found

## Troubleshooting

### FAISS Index Error
If you see "Error in read_index", the index is corrupted. It will automatically rebuild on next run.

### CSV Parsing Errors
Ensure the CSV uses semicolon delimiters and `latin-1` encoding for French characters.

### API Key Issues
Verify your HuggingFace API key is set in the `.env` file and has access to the Llama model.

## Future Improvements

- Add support for multiple languages
- Implement custom fine-tuning for domain-specific accuracy
- Add feedback mechanism to improve retrieval quality
- Support for document uploads
- Analytics dashboard for interaction data
- Caching of frequent queries

## License

This project is part of the Hackaton for the Pôle Léonard de Vinci.

## Contributors

- Arturo1s (Repository Owner)

## Contact

For support or questions, contact: guillaume.douceron@devinci.fr
