from dotenv import load_dotenv
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

from typing import Any, Dict, List
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

INDEX_NAME = "langchain-doc-index"


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    llm = ChatOllama(model="llama3.1")

    # the prompt considering the history
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    stuff_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    history_aware_retrieval = create_history_aware_retriever(
        llm=llm, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retrieval, combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }
    return new_result


if __name__ == "__main__":
    res = run_llm(query="what is LangChain chain?")
    print(res["result"])
