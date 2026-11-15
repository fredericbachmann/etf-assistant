"""The web front end: a chat page over the agent, served by FastAPI.

  config.py      settings from environment variables
  budget.py      the server's spending cap, shared by all conversations, reset daily
  documents.py   the source PDFs that citations link to
  citations.py   answer text -> Markdown with citations linked to PDF pages
  chat.py        agent events -> chat messages (no Gradio here, so it is testable alone)
  ui.py          the Gradio page: layout and wiring only
  app.py         FastAPI: health check, PDF route, login, the page; `uv run web`
"""
