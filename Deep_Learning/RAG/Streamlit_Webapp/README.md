# RAG System with Streamlit UI

A professional Retrieval-Augmented Generation (RAG) system with a user-friendly Streamlit interface. This project leverages HuggingFace models to create a powerful document retrieval and question answering system.

## Features

- **Document Processing**: Upload and process documents to build a knowledge base
- **Semantic Search**: Retrieve relevant information using advanced embedding models
- **Question Answering**: Generate accurate answers based on retrieved documents
- **Interactive UI**: User-friendly interface for interacting with the RAG system

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/CanaryBrianArmas/Personal_Projects/tree/main/Deep_Learning/RAG/Streamlit_Webapp
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate # Depending on your system you'll have to use bin (Linux) or Scripts (Windows)
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your HuggingFace API token:
   ```
   HUGGINGFACE_API_TOKEN=your_token_here
   ```

## Usage

1. Start the Streamlit app:
   ```bash
   streamlit run app/main.py
   ```

2. Access the app in your browser at `http://localhost:8501`

3. Upload documents, ask questions, and explore the system's capabilities

## Project Structure

```
rag-system/
├── app/              # Streamlit UI and application code
├── rag/              # Core RAG functionality
└── tests/            # Unit and integration tests
```

## Development

Create feature branch:
git checkout -b feature/new-feature

Make changes and commit:
git add .
git commit -m "Add new feature"

Push to remote (when ready):
git push origin feature/new-feature

### Testing

Run tests with pytest:
```bash
pytest
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.