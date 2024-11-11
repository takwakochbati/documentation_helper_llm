import os
from dotenv import load_dotenv
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_pinecone import PineconeVectorStore


load_dotenv()


def ingest_docs():

    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")
    raw_document = loader.load()
    # PineconeVectorStore.from_documents(texts, embeddings, index_name = os.environ['INDEX_NAME'])
    print(f"loaded {len(raw_document)} documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=15)
    documents = text_splitter.split_documents(raw_document)
    # print (documents[6])
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    PineconeVectorStore.from_documents(
        documents, embeddings, index_name="langchain-doc-index"
    )

    print("********loading to vectorstore done******")


if __name__ == "__main__":
    ingest_docs()
