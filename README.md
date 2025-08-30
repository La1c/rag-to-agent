# RAG → Agents Workshop

Hands-on workshop repo for evolving a basic Retrieval-Augmented Generation (RAG) system into a tool-using research agent. You’ll build and run a local retrieval pipeline, wire it into an agent via MCP tools, and explore orchestration with LangGraph and structured prompting.


## Prerequisites
- Python 3.13 (recommended, matches pinned `requirements.txt`)
- macOS or Linux; Windows via WSL is fine
- docker compose to run Langfuse


## Setup
1) Create and activate a virtual environment
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

2) Install dependencies
```
pip install -r requirements.txt
```

Notes:
- The first run may download model weights (sentence-transformers). Ensure internet access.
- `requirements.txt` is pin-generated from `requirements.in` (via pip-compile) for Python 3.13.

3) Configure environment variables
- Copy `env-template` to `.env` and fill the values:
  - `OPENAI_API_KEY`: required for LLM calls
  - `TAVILY_API_KEY`: required for web search tools
  - `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`: optional for tracing

4) Prepare the local vector store (offline retrieval)
- Ensure the `chroma_30K/` folder exists. If missing, unzip the archive in the repo root:
```
unzip chroma_30K.zip -d .
```

5) GPU/MPS device note
- In `mcp_retrieval.py`, embeddings default to `device="mps"` (Apple Silicon). If you’re on CPU or CUDA, change the `device` argument to `"cpu"` or `"cuda"` accordingly.


## Conda Setup (environment.yaml)
If you prefer Conda over `venv`, you can create the environment from `environment.yaml`.

- Create the environment (Mamba recommended for speed):
```
conda env create -f environment.yaml
```

- Activate it:
```
conda activate rag-workshop
```

- Update after changes to `environment.yaml`:
```
conda env update -f environment.yaml --prune
```

## Running the Notebook
- Open `rag_to_agent_workshop.ipynb` in VS Code or any Jupyter frontend.
- Select the  kernel and run cells top-to-bottom.
- The notebook demonstrates:
  - Pydantic prompts for clarification, planning, and summarization
  - LangGraph workflows with tool-calling
  - Using an MCP client to call the local retrieval tool

MCP integration: The notebook uses `langchain_mcp_adapters` and a `MultiServerMCPClient` that can launch the retrieval server via module mode (`-m mcp_retrieval`). You don’t need to start it manually if you follow the notebook cells.

## Project Structure
- `rag_to_agent_workshop.ipynb`: main, end-to-end workshop notebook demonstrating RAG → agents.
- `mcp_retrieval.py`: MCP server exposing a local `Chroma` retriever using `Snowflake/snowflake-arctic-embed-m-v1.5` embeddings.
- `prompts.py`: PydanticPrompt classes for clarification, planning, summarization; includes example schemas and prompt templates used in the workshop.
- `tools.py`: research utilities and LangChain tools (`tavily_search`, `think_tool`, `ConductResearch`, `ResearchComplete`).
- `requirements.in` / `requirements.txt`: dependency sources and pinned lockfile (3.13).
- `env-template`: sample env vars for `.env` (`OPENAI_API_KEY`, `TAVILY_API_KEY`, Langfuse keys).
- `chroma_30K/` and `chroma_30K.zip`: persisted local vector store with ~30K wiki chunks.
- `resources/`: images/diagrams referenced by the notebook.


## Tips & Troubleshooting
- Large downloads: first-time model and tokenizer fetches can take a while.
- Tavily usage: web search features require `TAVILY_API_KEY`; the local MCP retriever does not.
- Langfuse: set keys and host in `.env` if you want tracing; otherwise it’s optional.
