## WEEK 4 DELIVERABLES

To run the test calls:
``` bash
source .venv/bin/activate
python3 ./week4/test-calls.py
```

## WEEK 5 DELIVERABLES

A small RAG experiment over 20 recent arXiv abstracts.

To run it:
``` bash
source .venv/bin/activate
python3 ./week5/fetch_corpus.py    # once: rebuilds the pinned arXiv snapshot
python3 ./week5/rag_experiment.py
```