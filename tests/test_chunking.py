"""Chunking rules (corpus/chunking.py), measured in characters so no model is needed."""

from datetime import date

from etf_assistant.corpus.chunking import chunk_pages, heading_text

META = dict(isin="IE0000000001", fund_name="Test Fund", doc_type="kid",
            as_of_date=date(2026, 9, 30), source_file="test.pdf")


def test_never_crosses_a_page():
    chunks = chunk_pages(["short page one", "short page two"], **META)
    assert [(c.page, c.text) for c in chunks] == [(1, "short page one"), (2, "short page two")]


def test_heading_starts_new_chunk_once_current_is_big_enough():
    body = "x" * 50
    page = f"{body}\n\n## Costs\n\nfee 0.20%"
    assert len(chunk_pages([page], **META, min_len=10)) == 2  # 50 chars >= min_len: split at the heading
    assert len(chunk_pages([page], **META, min_len=100)) == 1  # 50 chars < min_len: heading joins the chunk


def test_section_carries_over_to_the_next_page():
    chunks = chunk_pages(["## Risk indicator\n\nSRI 4", "continued text"], **META)
    assert chunks[1].section == "Risk indicator"


def test_oversized_block_is_split_by_sentences():
    block = " ".join(f"Sentence number {i}." for i in range(40))
    chunks = chunk_pages([block], **META, max_len=200)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


def test_context_header_names_fund_document_and_page():
    c = chunk_pages(["## Costs\n\nfee 0.20%"], **META)[0]
    assert c.embedding_text().startswith("Test Fund (IE0000000001) | Key Information Document (KID), page 1")


def test_heading_text_strips_bold_markers():
    assert heading_text("## **Risk Indicator**") == "Risk Indicator"
    assert heading_text("plain paragraph") is None
