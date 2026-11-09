#!/usr/bin/env python3
"""Week 5: a small RAG experiment over recent arXiv abstracts.

Compares the same chat model with no context (baseline) against the same model
given the top-3 retrieved abstracts (RAG).

Run:
    python week5/rag_experiment.py --limit 5 --no-baseline   # contamination pilot
    python week5/rag_experiment.py                           # full run
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
API = "https://openrouter.ai/api/v1"

# Free endpoints are blocked by this workspace's zero-data-retention guardrail,
# so the defaults are low-cost paid models. Override with --chat-model/--embed-model.
CHAT_MODEL = "google/gemini-2.5-flash"
EMBED_MODEL = "openai/text-embedding-3-small"
TOP_K = 3
CHUNK_WORDS = 120
OVERLAP = 30
TEMPERATURE = 0.0
MAX_TOKENS = 200
SLEEP_BETWEEN_QUESTIONS = 1.0

HEADERS = {
    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
    "Content-Type": "application/json",
}

REFUSALS = (
    "i don't know",
    "i do not know",
    "not mentioned",
    "no information",
    "cannot determine",
    "not provided",
    "not available",
)


def post_with_retry(path, payload, attempts=5):
    """POST to OpenRouter, retrying rate limits and transient server errors."""
    for attempt in range(attempts):
        response = requests.post(
            API + path, headers=HEADERS, json=payload, timeout=120
        )
        if response.status_code == 200:
            return response.json()
        if response.status_code in (429, 500, 502, 503, 504):
            wait = 2 ** attempt
            print(f"HTTP {response.status_code} from {path}; retrying in {wait}s")
            time.sleep(wait)
            continue
        print(f"HTTP {response.status_code}: {response.text[:500]}")
        response.raise_for_status()

    raise RuntimeError(f"failed after {attempts} attempts: {path}")


def embed(texts, model=EMBED_MODEL):
    """Return L2-normalized embeddings, preserving input order."""
    data = post_with_retry("/embeddings", {"model": model, "input": texts})
    ordered = sorted(data["data"], key=lambda item: item["index"])
    vectors = np.array([item["embedding"] for item in ordered], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def load_json(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def load_questions(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def validate(docs, questions):
    doc_ids = {doc["id"] for doc in docs}
    if len(doc_ids) != len(docs):
        raise ValueError("duplicate document ids")
    if any(not doc.get("text") for doc in docs):
        raise ValueError("empty document text")

    seen = set()
    for question in questions:
        qid = question["id"]
        if qid in seen:
            raise ValueError(f"duplicate question id: {qid}")
        seen.add(qid)

        unknown = set(question["relevant_doc_ids"]) - doc_ids
        if unknown:
            raise ValueError(f"{qid}: unknown doc ids {sorted(unknown)}")
        if question["answerable"] and not question["answer_key"]:
            raise ValueError(f"{qid}: answerable question has no answer_key")
        if not question["answerable"] and question["relevant_doc_ids"]:
            raise ValueError(f"{qid}: unanswerable question has relevant docs")
        if question.get("match", "any") not in ("any", "all"):
            raise ValueError(f"{qid}: match must be 'any' or 'all'")


def chunk_text(text, size=CHUNK_WORDS, overlap=OVERLAP):
    words = text.split()
    if len(words) <= size:
        return [text]
    step = size - overlap
    return [" ".join(words[i : i + size]) for i in range(0, len(words) - overlap, step)]


def retrieve(query_vec, chunk_vecs, chunks, k=TOP_K):
    """Top-k distinct documents: walk the ranking, keep the first chunk per doc."""
    scores = chunk_vecs @ query_vec
    picked = []
    seen_docs = set()
    for index in np.argsort(scores)[::-1]:
        chunk = chunks[index]
        if chunk["doc_id"] in seen_docs:
            continue
        seen_docs.add(chunk["doc_id"])
        picked.append((chunk, float(scores[index])))
        if len(picked) == k:
            break
    return picked


def answer(question, context=None, model=CHAT_MODEL):
    if context:
        system = (
            "Answer the question using ONLY the context below. "
            "If the answer is not in the context, reply exactly: I don't know.\n\n"
            f"Context:\n{context}"
        )
    else:
        system = "Answer the question concisely."

    data = post_with_retry(
        "/chat/completions",
        {
            "model": model,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
        },
    )
    return (data["choices"][0]["message"]["content"] or "").strip()


def is_correct(text, item):
    lowered = text.lower()
    if not item["answerable"]:
        return any(refusal in lowered for refusal in REFUSALS)

    keys = [key.lower() for key in item["answer_key"]]
    if item.get("match", "any") == "all":
        return all(key in lowered for key in keys)
    return any(key in lowered for key in keys)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions", default=str(DATA / "questions.jsonl"), help="JSONL file"
    )
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    parser.add_argument(
        "--no-baseline", action="store_true", help="skip the no-context baseline"
    )
    parser.add_argument("--output", default=str(BASE / "results.csv"))
    parser.add_argument("--chat-model", default=CHAT_MODEL)
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    args = parser.parse_args()

    docs = load_json("corpus.json")
    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    validate(docs, questions)

    docs_by_id = {doc["id"]: doc for doc in docs}
    chunks = [
        {
            "chunk_id": f"{doc['id']}-c{i}",
            "doc_id": doc["id"],
            "text": piece,
            "embed_text": f"{doc['title']}. {piece}",
        }
        for doc in docs
        for i, piece in enumerate(chunk_text(doc["text"]))
    ]
    print(f"{len(docs)} docs -> {len(chunks)} chunks; {len(questions)} questions")
    print(f"chat={args.chat_model}  embed={args.embed_model}")

    chunk_vecs = embed([chunk["embed_text"] for chunk in chunks], args.embed_model)
    question_vecs = embed(
        [question["question"] for question in questions], args.embed_model
    )

    rows = []
    for question, question_vec in zip(questions, question_vecs):
        hits = retrieve(question_vec, chunk_vecs, chunks)
        retrieved_docs = [chunk["doc_id"] for chunk, _ in hits]
        context = "\n\n".join(
            f"[{doc_id}] {docs_by_id[doc_id]['title']}\n"
            f"{docs_by_id[doc_id]['text']}"
            for doc_id in retrieved_docs
        )

        baseline = (
            ""
            if args.no_baseline
            else answer(question["question"], model=args.chat_model)
        )
        rag = answer(question["question"], context, model=args.chat_model)

        relevant = set(question["relevant_doc_ids"])
        hit_at_3 = (
            int(bool(relevant & set(retrieved_docs))) if question["answerable"] else ""
        )
        recall_at_3 = (
            len(relevant & set(retrieved_docs)) / len(relevant) if relevant else ""
        )

        rows.append(
            {
                "id": question["id"],
                "type": question["type"],
                "question": question["question"],
                "relevant_doc_ids": ";".join(question["relevant_doc_ids"]),
                "retrieved_doc_ids": ";".join(retrieved_docs),
                "hit_at_3": hit_at_3,
                "recall_at_3": recall_at_3,
                "baseline_answer": baseline,
                "baseline_correct": ""
                if args.no_baseline
                else int(is_correct(baseline, question)),
                "rag_answer": rag,
                "rag_correct": int(is_correct(rag, question)),
            }
        )
        time.sleep(SLEEP_BETWEEN_QUESTIONS)

    with Path(args.output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    answerable = [row for row in rows if row["type"] != "unanswerable"]
    unanswerable = [row for row in rows if row["type"] == "unanswerable"]

    print(f"hit@3      : {mean([row['hit_at_3'] for row in answerable]):.2f}")
    print(f"recall@3   : {mean([row['recall_at_3'] for row in answerable]):.2f}")
    if not args.no_baseline:
        print(f"baseline   : {mean([row['baseline_correct'] for row in answerable]):.2f}")
    print(f"RAG        : {mean([row['rag_correct'] for row in answerable]):.2f}")
    if unanswerable:
        if not args.no_baseline:
            print(
                "refusals   : baseline "
                f"{mean([row['baseline_correct'] for row in unanswerable]):.2f} "
                f"vs RAG {mean([row['rag_correct'] for row in unanswerable]):.2f}"
            )
        else:
            print(
                "refusals   : RAG "
                f"{mean([row['rag_correct'] for row in unanswerable]):.2f}"
            )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
