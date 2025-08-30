from typing import Any

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.embeddings.embeddings import Embeddings
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

mcp = FastMCP("local-retrieval")


class ArcticEmbeddings(Embeddings):
    def __init__(self, model_name, batch_size=32):
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=device,
            model_kwargs={"torch_dtype": "float16"},
        )
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        return self.model.encode(texts, batch_size=self.batch_size)

    def embed_query(self, text: str) -> list[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        return self.model.encode(text, batch_size=self.batch_size, prompt="query")


class ToolClient:
    def __init__(self):
        model_name = "Snowflake/snowflake-arctic-embed-m-v1.5"
        chroma_client = Chroma(
            collection_name="wiki_chunks_30K",
            persist_directory="./chroma_30K",
            embedding_function=ArcticEmbeddings(model_name),
            client_settings=Settings(anonymized_telemetry=False),
        )

        self.retriever = chroma_client.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 20, "fetch_k": 50, "lambda_mult": 0.8},
        )


_tool_client_instance = None


def get_tool_client():
    global _tool_client_instance
    if _tool_client_instance is None:
        _tool_client_instance = ToolClient()
    return _tool_client_instance


class DocModel(BaseModel):
    content: str
    url: str | None = None
    title: str | None = None


@mcp.tool()
async def retrieve_docs(query: str, top_k: int = 3) -> list[DocModel]:
    """
    Retrieve relevant passages from a local, offline vector index — not the web.

    This tool searches only your locally indexed documents (Chroma) and never
    accesses the internet. Results are semantic matches from your curated corpus,
    not live web pages or news. Use web search tools for external or up-to-date
    information.

    Args:
        query (str): Natural-language query to search the local index.
        top_k (int, optional): Maximum number of passages to return. Defaults to 3.

    Returns:
        list[dict[str, Any]]: Each item includes 'content' (text) and may include
        'url' and 'title' when available.
    """
    retriever = get_tool_client().retriever
    results = await retriever.ainvoke(query, k=top_k)
    return [
        DocModel(
            content=doc.page_content,
            url=doc.metadata.get("url"),
            title=doc.metadata.get("title"),
        )
        for doc in results
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")
