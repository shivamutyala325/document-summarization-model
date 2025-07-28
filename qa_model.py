import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline


def build_rag_pipeline(file_path):
    # 1. Load the document
    print("Loading document...")
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # 2. Split the document into chunks
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    # 3. Create embeddings and store in a local vector database (FAISS)
    print("Creating embeddings and vector store...")
    embedding_model = HuggingFaceEmbeddings(model_name="embedding_model")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)

    return vectorstore


def qanda():
    file_path = input('paste the file path here: ')
    vectorstore = build_rag_pipeline(file_path)
    retriever = vectorstore.as_retriever()

    print("Loading local Question-Answering model...")
    qa_model_name = "QandA_model"
    qa_pipeline = pipeline("question-answering", model=qa_model_name, tokenizer=qa_model_name)

    print("\n--- Document Chatbot Ready ---")
    print("Ask questions about the document. Type 'exit' to quit.")

    while True:
        question = input("\nYour question: ")
        if question.lower() == 'exit':
            break

        print("Finding answer...")

        docs = retriever.invoke(question)
        context = " ".join([doc.page_content for doc in docs])
        result = qa_pipeline(question=question, context=context)
        answer = result.get('answer', "Sorry, I couldn't find an answer.")

        print(f"\nAnswer: {answer}")

qanda()