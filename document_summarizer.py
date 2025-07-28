import sys
from transformers import pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader


print("Loading local summarization model...")
local_model_path = "./text_summarizer_model"
summarizer_pipeline = pipeline("summarization", model=local_model_path, tokenizer=local_model_path)
llm = HuggingFacePipeline(pipeline=summarizer_pipeline)

chain = load_summarize_chain(llm, chain_type="map_reduce")

def summarize_document(file_path):
    try:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            return "Error: Unsupported file type."

        docs = loader.load()

        print("Document loaded. Generating summary with LangChain...")
        summary_output = chain.invoke(docs)

        return summary_output['output_text']

    except Exception as e:
        return f"An error occurred: {e}"

document_path=input('enter your document path')

final_summary = summarize_document(document_path)

print(" DOCUMENT SUMMARY With LangChain")
print(final_summary)
