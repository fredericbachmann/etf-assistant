"""The chat page: Gradio layout, look and wiring, nothing else.

What an answer looks like is decided in chat.py; which agent, budget and documents
are used is decided in app.py, which passes them in as factories.

Each browser session has its own Agent (held in gr.State, dropped after STATE_TTL_S).
While an answer is being written, the input is disabled: two questions at once would
interleave in the same conversation history.
"""

from collections.abc import Callable

import gradio as gr

from etf_assistant.agent.loop import Agent
from etf_assistant.web.chat import Transcript, respond

STATE_TTL_S = 3 * 3600

HEADER = """
<div class="app-title">ETF Assistant</div>
<div class="app-subtitle">Answers from the funds' KIDs and factsheets, with every figure cited.</div>
"""
DISCLAIMER = ("Information only, not investment advice. Figures are as of the documents' dates. "
              "Click a citation to open the source page.")

EXAMPLES = [
    "What's the ongoing charge of VWCE?",
    "Which fund has the lowest TER?",
    "Compare the risk indicators of all funds.",
    "How much would fees cost on €10,000 over 20 years in EUNL?",
]

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    text_size=gr.themes.sizes.text_md,
    radius_size=gr.themes.sizes.radius_md,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="*neutral_50",
    body_background_fill_dark="*neutral_950",
    block_shadow="none",
    block_border_width="1px",
)

CSS = """
.gradio-container { max-width: 860px !important; margin: 0 auto !important; }
footer { display: none !important; }

#header { align-items: center; padding: 8px 0 4px; }
#header .app-title { font-size: 1.4rem; font-weight: 600; letter-spacing: -0.015em; }
#header .app-subtitle { margin-top: 2px; font-size: 0.92rem; color: var(--body-text-color-subdued); }

#chat a[href^="/documents/"] {
  color: var(--color-accent); text-decoration: none; font-weight: 500; white-space: nowrap;
}
#chat a[href^="/documents/"]:hover { text-decoration: underline; }

#chat .placeholder-content { padding: 32px 24px; }
#chat .placeholder { margin-bottom: 12px; color: var(--body-text-color-subdued); }
#chat .examples { display: grid !important; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
#chat .examples .example { width: auto !important; max-width: none !important; }
@media (max-width: 600px) { #chat .examples { grid-template-columns: 1fr; } }

#disclaimer, #disclaimer * {
  text-align: center; font-size: 0.78rem !important; color: var(--body-text-color-subdued) !important;
}
#disclaimer p { margin: 0; }
"""

# Citation links open the PDF in a new tab: leaving the page would end the conversation.
HEAD = """
<script>
document.addEventListener("click", (e) => {
  const a = e.target.closest && e.target.closest('a[href^="/documents/"]');
  if (a) { e.preventDefault(); window.open(a.href, "_blank", "noopener"); }
}, true);
</script>
"""


def fund_table(funds: list[tuple[str, str, list[str]]]) -> str:
    """(name, ISIN, tickers) per fund, as a Markdown table."""
    rows = [f"| {name} | {', '.join(tickers) or '–'} | {isin} |" for name, isin, tickers in funds]
    return "\n".join(["| Fund | Tickers | ISIN |", "|---|---|---|", *rows])


def build_ui(*, new_agent: Callable[[], Agent], new_transcript: Callable[[], Transcript],
             funds: list[tuple[str, str, list[str]]], max_question_chars: int, concurrency: int) -> gr.Blocks:
    with gr.Blocks(title="ETF Assistant", analytics_enabled=False) as ui:
        with gr.Row(elem_id="header"):
            gr.HTML(HEADER)
            new = gr.Button("New conversation", variant="secondary", size="sm", scale=0, min_width=150)

        with gr.Accordion(f"Funds in the library ({len(funds)})", open=False):
            gr.Markdown(fund_table(funds))

        chatbot = gr.Chatbot(elem_id="chat", show_label=False, height="62vh", buttons=["copy_all"],
                             feedback_options=None, layout="panel",
                             latex_delimiters=[],  # "$" in amounts is money, not maths
                             examples=[{"text": q} for q in EXAMPLES],
                             placeholder="**Ask about a fund, or compare several.**")
        agent = gr.State(None, time_to_live=STATE_TTL_S)
        question = gr.Textbox(show_label=False, placeholder="e.g. What's the TER of EUNL?",
                              max_length=max_question_chars, submit_btn=True, autofocus=True)
        gr.Markdown(DISCLAIMER, elem_id="disclaimer")

        def answer(text: str, history: list, agent: Agent | None):
            text = text.strip()
            if not text:
                yield gr.skip(), gr.skip(), gr.skip(), gr.skip()
                return
            agent = agent or new_agent()
            history = [*history, {"role": "user", "content": text}]
            messages = []
            yield gr.Textbox(value="", interactive=False), gr.Button(interactive=False), history, agent
            for messages in respond(agent, text, new_transcript()):
                yield gr.skip(), gr.skip(), history + messages, agent
            yield gr.Textbox(interactive=True), gr.Button(interactive=True), history + messages, agent

        def pick_example(evt: gr.SelectData) -> str:
            return evt.value["text"]

        ask = dict(fn=answer, inputs=[question, chatbot, agent], outputs=[question, new, chatbot, agent],
                   concurrency_limit=concurrency)
        question.submit(**ask)
        chatbot.example_select(pick_example, None, question).then(**ask)
        # the Chatbot's own clear (trash) button must forget the conversation too, not just hide it
        new.click(lambda: ([], None), None, [chatbot, agent])
        chatbot.clear(lambda: None, None, agent)
    return ui
