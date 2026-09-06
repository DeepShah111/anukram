# Two-stage answer generation: a cited draft from the generator, then an auditor that strips unsupported claims.

import os
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import (
    logger, GENERATOR_MODEL, MAX_API_RETRIES, API_RETRY_MIN_WAIT, API_RETRY_MAX_WAIT,
)

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

GENERATOR_PROMPT = """You are a legal case assistant for Indian police and courts.
Answer the QUESTION using ONLY the CONTEXT below. After every fact, cite its source as
[case_id, page]. If the context does not contain the answer, say so plainly.
Do not invent names, numbers, or sections.

CONTEXT:
{context}

QUESTION: {question}

Answer with inline [case_id, page] citations:"""

AUDITOR_PROMPT = """You are an evidentiary auditor. Check the DRAFT answer against the CONTEXT.
Remove or correct any sentence that is not directly supported by the context.
Keep every valid [case_id, page] citation. Return only the cleaned answer.

CONTEXT:
{context}

DRAFT:
{draft}

Cleaned, fully-supported answer:"""


def _format_context(docs):
    # Turn retrieved chunks into a numbered context block with case id and page.
    blocks = []
    for doc in docs:
        m = doc.metadata
        blocks.append(f"[{m.get('case_id')}, page {m.get('page')}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


@retry(stop=stop_after_attempt(MAX_API_RETRIES),
       wait=wait_exponential(min=API_RETRY_MIN_WAIT, max=API_RETRY_MAX_WAIT))
def _call_llm(prompt):
    # Single Groq chat completion with retry on transient failures.
    resp = _client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def generate_answer(question, retrieved_docs):
    # Draft a cited answer, then audit it, and return both stages plus the sources used.
    context = _format_context(retrieved_docs)

    draft = _call_llm(GENERATOR_PROMPT.format(context=context, question=question))
    audited = _call_llm(AUDITOR_PROMPT.format(context=context, draft=draft))

    sources = [{"case_id": d.metadata.get("case_id"), "page": d.metadata.get("page")}
               for d in retrieved_docs]
    logger.info("Answer generated and audited for question: %s", question[:60])
    return {"question": question, "draft": draft, "answer": audited, "sources": sources}


if __name__ == "__main__":
    from src.synthetic_corpus_generator import generate_corpus
    from src.document_ingestion import load_and_chunk_documents
    from src.retrieval_engine import HybridRetrievalEngine

    generate_corpus()
    chunks = load_and_chunk_documents()
    retriever = HybridRetrievalEngine().build(document_chunks=chunks, rebuild=False)

    docs = retriever.invoke("Which cases mention vehicle DL-01-AB-1234?")
    result = generate_answer("Which cases mention vehicle DL-01-AB-1234?", docs)
    print("\nANSWER:\n", result["answer"])