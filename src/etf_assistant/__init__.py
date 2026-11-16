"""ETF fine-print assistant: answers questions about ETFs from their KIDs and factsheets.

Layers, each importing only from the ones above it:
  paths, llm   shared infrastructure (file locations; Claude client, cost and budget)
  corpus       the documents: manifest, PDF parsing, chunking, fund catalog
  retrieval    embeddings and the OpenSearch index
  pipeline     data setup: ingest (PDFs -> chunks) and index (chunks -> OpenSearch)
  agent        the assistant: prompt, tools, agentic loop, CLI
  web          the chat page: FastAPI + Gradio over the agent
  evals        answer key, golden questions, retrieval and end-to-end evals

tests/test_layers.py holds the exact rule per layer.
"""
