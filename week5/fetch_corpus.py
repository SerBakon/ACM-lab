#!/usr/bin/env python3
"""Fetch a pinned snapshot of 20 arXiv abstracts and build the week 5 corpus.

Run once:  python week5/fetch_corpus.py

The experiment itself reads only the committed snapshot, so it can run offline.
"""

import hashlib
import json
import time
from datetime import date
from pathlib import Path
import xml.etree.ElementTree as ET

import requests

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RAW = DATA / "raw"

# Recent retrieval-augmented generation papers (Nov 2026). Frozen after the
# first successful fetch; see data/sources.json for the exact versions.
ARXIV_IDS = [
    "2611.09807",
    "2611.09791",
    "2611.09658",
    "2611.09423",
    "2611.09414",
    "2611.09298",
    "2611.09232",
    "2611.09090",
    "2611.08935",
    "2611.08054",
    "2611.07928",
    "2611.07517",
    "2611.07393",
    "2611.06715",
    "2611.06498",
    "2611.06481",
    "2611.05900",
    "2611.05596",
    "2611.04625",
    "2611.04264",
]

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}


def fetch(ids, attempts=5):
    """Download the Atom feed for the pinned IDs, retrying transient errors."""
    params = {"id_list": ",".join(ids), "max_results": len(ids)}
    headers = {"User-Agent": "ucsc-acm-lab-week5/1.0"}
    for attempt in range(attempts):
        try:
            response = requests.get(API, params=params, headers=headers, timeout=60)
        except requests.RequestException as exc:
            if attempt == attempts - 1:
                raise
            wait = 2 ** attempt
            print(f"network error ({exc}); retrying in {wait}s")
            time.sleep(wait)
            continue

        if response.status_code == 200:
            return response.text
        if response.status_code in (429, 503):
            wait = 2 ** attempt
            print(f"{response.status_code} from arXiv; retrying in {wait}s")
            time.sleep(wait)
            continue
        response.raise_for_status()

    raise RuntimeError("arXiv fetch failed after retries")


def parse(xml_text):
    """Parse Atom entries into flat records."""
    root = ET.fromstring(xml_text)
    entries = []
    for entry in root.findall("a:entry", NS):
        arxiv_id = entry.find("a:id", NS).text.rsplit("/abs/", 1)[1]
        title = " ".join((entry.find("a:title", NS).text or "").split())
        abstract = " ".join((entry.find("a:summary", NS).text or "").split())
        published = (entry.find("a:published", NS).text or "")[:10]
        if not abstract:
            continue
        entries.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "text": abstract,
                "published": published,
            }
        )
    return entries


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    xml_text = fetch(ARXIV_IDS)
    (RAW / "arxiv.xml").write_text(xml_text, encoding="utf-8")

    entries = parse(xml_text)
    if len(entries) != len(ARXIV_IDS):
        print(f"warning: requested {len(ARXIV_IDS)} ids, parsed {len(entries)} entries")

    docs = []
    sources = []
    for index, entry in enumerate(entries, start=1):
        doc_id = f"doc{index:02d}"
        url = f"https://arxiv.org/abs/{entry['arxiv_id']}"
        docs.append(
            {
                "id": doc_id,
                "arxiv_id": entry["arxiv_id"],
                "title": entry["title"],
                "text": entry["text"],
                "published": entry["published"],
                "url": url,
            }
        )
        sources.append(
            {
                "doc_id": doc_id,
                "arxiv_id": entry["arxiv_id"],
                "url": url,
                "license": "see arXiv abs page",
                "sha256": hashlib.sha256(entry["text"].encode("utf-8")).hexdigest(),
            }
        )

    (DATA / "corpus.json").write_text(
        json.dumps(docs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (DATA / "sources.json").write_text(
        json.dumps(
            {
                "retrieved": str(date.today()),
                "source": "arXiv API",
                "query": "retrieval-augmented generation, Nov 2026",
                "documents": sources,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"wrote {len(docs)} docs to {DATA / 'corpus.json'}")


if __name__ == "__main__":
    main()
