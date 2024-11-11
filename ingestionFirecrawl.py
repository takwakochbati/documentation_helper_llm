from langchain_community.document_loaders import FireCrawlLoader
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_pinecone import PineconeVectorStore
load_dotenv()

def ingest_doc_firecrawl():

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    langchain_documents_base_urls = [
        "https://python.langchain.com/v0.2/docs/integrations/chat/",
        "https://python.langchain.com/v0.2/docs/integrations/llms/",
        "https://python.langchain.com/v0.2/docs/integrations/text_embedding/",
        "https://python.langchain.com/v0.2/docs/integrations/document_loaders/",
        "https://python.langchain.com/v0.2/docs/integrations/document_transformers/",
        "https://python.langchain.com/v0.2/docs/integrations/vectorstores/",
        "https://python.langchain.com/v0.2/docs/integrations/retrievers/",
        "https://python.langchain.com/v0.2/docs/integrations/tools/",
        "https://python.langchain.com/v0.2/docs/integrations/stores/",
        "https://python.langchain.com/v0.2/docs/integrations/llm_caching/",
        "https://python.langchain.com/v0.2/docs/integrations/graphs/",
        "https://python.langchain.com/v0.2/docs/integrations/memory/",
        "https://python.langchain.com/v0.2/docs/integrations/callbacks/",
        "https://python.langchain.com/v0.2/docs/integrations/chat_loaders/",
        "https://python.langchain.com/v0.2/docs/concepts/",
    ]
    langchain_documents_base_urls2 = [langchain_documents_base_urls[0]]
    for url in langchain_documents_base_urls2:
        print(f'Crawling URL {url=}')
        loader = FireCrawlLoader(
            url = url,
            mode="crawl",
            params={"limit":5,
                    },
        )
        docs = loader.load()
        print(f"Going to add {len(docs)} documents to Pinecone")
        PineconeVectorStore.from_documents(docs, embeddings, index_name = "firecrawl-index")

        print(f"*****Loading {url}* to vectorstore done******")

if __name__ == "__main__":
    ingest_doc_firecrawl()
