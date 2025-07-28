# Document Summarizer & QnA Chatbot with RAG

This project includes two main functionalities:
Document Summarization, Q&A Chatbot Based on Document Context

 1. Document Summarization

#Files Involved:
- `download_summariser_model.py`
- `document_summarizer.py`

# Description:
This module allows you to summarize a document using a locally downloaded Hugging Face model.

# How to Use:
1. **Download the summarization model**:
   ```bash
   python download_summariser_model.py
   ```

2. **Run the summarizer**:
   ```bash
   python document_summarizer.py
   ```

---

 2. Q&A Chatbot with RAG

# Files Involved:
- `download_qa_model.py`
- `download_embedding_model.py`
- `qa_model.py`

# Description:
This module allows you to upload a document and interact with a chatbot that can answer questions based on the content of that document.

#How to Use:
1. **Download the QnA model**:
   ```bash
   python download_qa_model.py
   ```

2. **Download the embedding model**:
   ```bash
   python download_embedding_model.py
   ```

3. **Run the chatbot**:
   ```bash
   python qa_model.py
   ```

---



#Notes

- All models are downloaded locally for offline use.
- Ensure you have internet access during the initial model downloads, Once the models are downloaded locally you can use them offline.

---



