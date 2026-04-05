# Basic Local RAG Backend

Production-clean MVP Python backend for a local Retrieval-Augmented Generation system powered by a locally running [Ollama](https://ollama.com/) instance. It ingests local files, embeds and stores chunks persistently, retrieves relevant context, injects optional prompt profiles, optionally mixes in web results, and returns grounded answers with source attribution.

## Features

- FastAPI backend only, with no frontend and no Docker requirement.
- Uses Ollama's local HTTP API for chat and embeddings.
- Supports `.txt`, `.md`, and `.pdf` ingestion.
- Persistent local vector storage with ChromaDB and document metadata in SQLite.
- Role/profile loading from `.txt`, `.json`, `.yaml`, and `.yml`.
- Retrieval modes: `local_only`, `web_only`, and `hybrid`.
- Optional DuckDuckGo-backed web search through a pluggable `WebSearchProvider`.
- Multipart query endpoint with optional image passthrough for multimodal Ollama models.
- Short per-session memory with bounded recent turns.
- Source attribution and optional debug output.
- Unit and API tests that mock Ollama calls instead of requiring a live Ollama instance.

## Architecture Overview

The project root is the backend application:

```text
app/
  api/
  core/
  loaders/
  services/
  storage/
  utils/
data/
  chroma/
  uploads/
  roles/
tests/
```

Key pieces:

- `app/services/ollama_client.py`: thin async HTTP wrapper around Ollama.
- `app/services/document_service.py`: file validation, persistence, extraction, chunking, embedding, indexing.
- `app/services/retrieval_service.py`: local vector retrieval and optional web retrieval.
- `app/services/rag_service.py`: prompt assembly, role injection, multimodal forwarding, session history, answer shaping.
- `app/storage/vector_store.py`: ChromaDB persistence.
- `app/storage/metadata_store.py`: SQLite metadata for ingested files.
- `app/loaders/role_loader.py`: startup/runtime loading of role files from disk.

## Requirements

- Python 3.11+
- Ollama installed locally and running
- A chat model pulled into Ollama
- An embedding model pulled into Ollama

## Installation

1. Create a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Copy the example environment file.

```bash
cp .env.example .env
```

On Windows:

```powershell
Copy-Item .env.example .env
```

## Running Ollama

If Ollama is not already serving locally, start it:

```bash
ollama serve
```

Pull a default chat model and embedding model:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

If you want multimodal image support, also pull a model you configured as `text_image`, for example:

```bash
ollama pull llava
```

The backend uses Ollama over HTTP at `http://localhost:11434/api` by default.

## Running the Backend

Start the API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://localhost:8000/api/v1/health
```

## Configuration

Configuration is driven by environment variables or `.env`.

Important settings:

- `OLLAMA_BASE_URL`
- `OLLAMA_CHAT_MODEL`
- `OLLAMA_EMBED_MODEL`
- `OLLAMA_KEEP_ALIVE`
- `OLLAMA_TIMEOUT_SECONDS`
- `OLLAMA_THINK`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `TOP_K`
- `SCORE_THRESHOLD`
- `DATA_DIR`
- `ROLES_DIR`
- `WEB_SEARCH_ENABLED`
- `WEB_SEARCH_PROVIDER`
- `SESSION_MAX_TURNS`
- `DEBUG_RAG`
- `MODEL_CAPABILITIES_JSON`

Example multimodal capability mapping:

```env
MODEL_CAPABILITIES_JSON={"llava":"text_image","gemma3":"text_image","gemma4":"text_image"}
```

Models not listed there are treated as `text_only`.

## Role Profiles

Roles are loaded from `data/roles/` on startup and can be reloaded at runtime.

Supported formats:

- `.txt`: entire file becomes `system_prompt`, filename becomes `name`
- `.json`
- `.yaml`
- `.yml`

Sample roles are included:

- `doctor.yaml`
- `lawyer.json`
- `coding_assistant.txt`

List roles:

```bash
curl http://localhost:8000/api/v1/roles
```

Reload roles from disk:

```bash
curl -X POST http://localhost:8000/api/v1/roles/reload
```

## Ingesting Documents

Upload one or more documents:

```bash
curl -X POST http://localhost:8000/api/v1/documents/ingest \
  -F "files=@/absolute/path/to/manual.txt" \
  -F "files=@/absolute/path/to/policy.pdf"
```

List ingested documents:

```bash
curl http://localhost:8000/api/v1/documents
```

Delete a document:

```bash
curl -X DELETE http://localhost:8000/api/v1/documents/<document_id>
```

Reindex a document:

```bash
curl -X POST http://localhost:8000/api/v1/documents/<document_id>/reindex
```

## Querying the System

`POST /api/v1/query` accepts `multipart/form-data` so it can also receive image files.

Example local-only query:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -F "question=Summarize the uploaded policy" \
  -F "retrieval_mode=local_only" \
  -F "role_name=coding_assistant" \
  -F "use_citations=true" \
  -F "debug=true"
```

Example hybrid query with web search enabled:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -F "question=Compare the local notes with recent public guidance" \
  -F "retrieval_mode=hybrid" \
  -F "use_citations=true"
```

Example multimodal query:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -F "question=Describe what is shown and relate it to the uploaded notes" \
  -F "chat_model=llava" \
  -F "retrieval_mode=local_only" \
  -F "images=@/absolute/path/to/image.png"
```

There is also an optional JSON endpoint for clients that prefer sending base64 images:

```bash
curl -X POST http://localhost:8000/api/v1/query/json \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does this screenshot show?",
    "chat_model": "llava",
    "retrieval_mode": "local_only",
    "image_base64": ["<base64-image-here>"]
  }'
```

Example response shape:

```json
{
  "answer": "Bananas are yellow [L1].",
  "sources": [
    {
      "source_id": "L1",
      "source_type": "local",
      "filename": "fruit.txt",
      "chunk_id": "123:0",
      "snippet_preview": "Bananas are yellow and rich in potassium.",
      "score": 0.93
    }
  ],
  "role_used": "coding_assistant",
  "retrieval_mode": "local_only",
  "session_id": "0db4f5f5-2d58-4b3d-9854-3d5166832f4e",
  "debug": {
    "prompt_preview": "...",
    "retrieved_local": [],
    "retrieved_web": [],
    "thinking": null,
    "model_timings": {}
  }
}
```

## Web Search

Web search is optional and disabled by default. Enable it in `.env`:

```env
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=duckduckgo
```

Notes:

- The default provider uses the `duckduckgo-search` Python package.
- Web results are normalized into `title`, `url`, `snippet`, and `source_type="web"`.
- In `hybrid` mode, local context and web context are kept separate in the prompt.
- If web search is disabled or unavailable, `web_only` and `hybrid` requests fail cleanly with a helpful error.

## Testing

Run the test suite:

```bash
pytest
```

The tests mock Ollama and do not require a running Ollama server.

## Limitations

- No OCR is performed. Images are forwarded directly to Ollama only when the selected chat model is configured as image-capable.
- Session memory is in-memory only and is scoped to the current backend process.
- Web search is intentionally simple and not hardened for production-scale search workloads.
- The MVP does not implement streaming responses.
- PDF extraction quality depends on the text layer present in the PDF.

## Future Improvements

- Add clean token streaming via SSE or NDJSON.
- Support more document loaders and richer metadata extraction.
- Add optional reranking and better prompt compression for larger knowledge bases.
- Support audio/video attachments behind the same multimodal abstraction.
- Add authentication and stronger rate limiting for multi-user deployments.
