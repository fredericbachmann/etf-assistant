"""The web server: FastAPI with the chat page mounted at /.

  GET /health              200 if the app and search work, 503 if search doesn't (for uptime checks)
  GET /documents/<file>    a source PDF, for citation links; only files listed in manifest.csv
  /                        the chat page (Gradio), behind a login if APP_PASSWORD is set

Run it with `uv run web`; settings come from environment variables (see config.py).
The PDF route is not behind the login: the documents are the issuers' public PDFs.
"""

import logging

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from etf_assistant.agent.loop import Agent
from etf_assistant.corpus.catalog import load_catalog
from etf_assistant.llm import warm_up
from etf_assistant.retrieval.store import check_ready
from etf_assistant.web import ui
from etf_assistant.web.budget import DailyBudget
from etf_assistant.web.chat import Transcript, describe_tool
from etf_assistant.web.citations import link_citations
from etf_assistant.web.config import Settings
from etf_assistant.web.documents import ROUTE, Documents

log = logging.getLogger(__name__)


def create_app(settings: Settings, documents: Documents) -> FastAPI:
    funds = load_catalog()
    fund_names = {isin: f.display_name for isin, f in funds.items()}
    budget = DailyBudget(settings.daily_budget_usd)  # one for all conversations

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    def health():
        problem = check_ready()
        return JSONResponse({"status": "degraded" if problem else "ok", "search": problem or "ok"},
                            status_code=503 if problem else 200)

    @app.get(ROUTE + "/{name}")
    def document(name: str):
        if not (path := documents.path_of(name)):
            raise HTTPException(404)
        return FileResponse(path, media_type="application/pdf", filename=path.name,
                            content_disposition_type="inline")

    def new_transcript() -> Transcript:
        return Transcript(link=lambda text: link_citations(text, documents.url_for),
                          describe=lambda name, args: describe_tool(name, args, fund_names))

    page = ui.build_ui(
        new_agent=lambda: Agent(budget=budget),
        new_transcript=new_transcript,
        funds=[(f.display_name, isin, f.tickers) for isin, f in funds.items()],
        max_question_chars=settings.max_question_chars,
        concurrency=settings.concurrency,
    )
    auth = (settings.username, settings.password) if settings.password else None
    # routes above take precedence over the page mounted at /
    return gr.mount_gradio_app(app, page, path="/", auth=auth, auth_message="ETF Assistant",
                               theme=ui.THEME, css=ui.CSS, head=ui.HEAD, footer_links=[])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("opensearch", "httpx", "httpx2"):  # one line per request
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = Settings.from_env()
    documents = Documents.from_manifest()
    if problem := check_ready():
        log.warning("search won't work: %s", problem)
    if not documents:
        log.warning("no source PDFs in data/: citations won't be links")
    if not settings.password and settings.host not in ("127.0.0.1", "localhost"):
        log.warning("APP_PASSWORD is not set: anyone who can reach %s can spend the budget", settings.host)
    try:
        warm_up()  # renew credentials once now rather than in several requests at once
    except Exception as e:
        log.warning("Claude API not reachable yet: %s", e)
    log.info("daily budget $%.2f, %d documents, http://%s:%d",
             settings.daily_budget_usd, len(documents), settings.host, settings.port)
    uvicorn.run(create_app(settings, documents), host=settings.host, port=settings.port)
