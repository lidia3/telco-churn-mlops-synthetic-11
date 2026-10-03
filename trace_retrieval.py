import json
import os
import re
from pathlib import Path
from uuid import uuid4

from fastembed import TextEmbedding
from langsmith import Client, traceable
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

if not os.environ.get("LANGSMITH_API_KEY"):
    raise SystemExit("LANGSMITH_API_KEY is missing. Enter the key first.")

os.environ["LANGSMITH_TRACING"] = "true"
os.environ.setdefault("LANGSMITH_PROJECT", "telco-churn-topic11")

ls = Client()
ls.read_projects(limit=1).__next__() if False else None
# Verify credentials before loading the model.
list(ls.list_projects(limit=1))
print("LangSmith connection OK", flush=True)

docs = json.loads(Path("data/documents.json").read_text())
by_id = {d["id"]: d for d in docs}
tests = json.loads(Path("data/eval_queries.json").read_text())

def tokens(text):
    return re.findall(r"\b\w+\b", text.lower())

bm25 = BM25Okapi([tokens(d["text"]) for d in docs])
qdrant = QdrantClient(url="http://localhost:6333")
embedder = TextEmbedding("BAAI/bge-small-en-v1.5", threads=2)

@traceable(name="embed_query", client=ls)
def embed_query(query):
    return next(embedder.query_embed(query)).tolist()

@traceable(name="qdrant_dense_search", run_type="retriever", client=ls)
def dense_search(vector):
    hits = qdrant.query_points(
        collection_name="churn_kb",
        query=vector,
        limit=10,
        with_payload=True,
    ).points
    return [
        {"id": int(h.id), "score": h.score, "text": h.payload["text"]}
        for h in hits
    ]

@traceable(name="bm25_search", client=ls)
def lexical_search(query):
    scores = bm25.get_scores(tokens(query))
    return [
        {"id": docs[i]["id"], "score": float(scores[i])}
        for i in sorted(
            range(len(docs)), key=lambda i: (-scores[i], docs[i]["id"])
        )
        if scores[i] > 0
    ][:10]

@traceable(name="rrf_fusion", client=ls)
def fuse(dense, lexical):
    scores = {}
    for results in (dense, lexical):
        for rank, item in enumerate(results, 1):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (60 + rank)
    ids = sorted(scores, key=lambda i: (-scores[i], i))[:3]
    return [
        {**by_id[i], "rrf_score": scores[i]}
        for i in ids
    ]

@traceable(name="retrieval_pipeline", client=ls)
def retrieve(query, segment, relevant_ids):
    vector = embed_query(query)
    dense = dense_search(vector)
    lexical = lexical_search(query)
    top3 = fuse(dense, lexical)
    hit = float(any(d["id"] in relevant_ids for d in top3))
    return {"documents": top3, "hit_at_3": hit, "segment": segment}

completed = []
for n, test in enumerate(tests, 1):
    run_id = uuid4()
    output = retrieve(
        test["query"],
        test["segment"],
        test["relevant_ids"],
        langsmith_extra={
            "run_id": run_id,
            "metadata": {
                "segment": test["segment"],
                "method": "hybrid",
                "synthetic": True,
            },
            "tags": ["topic11", test["segment"]],
        },
    )
    completed.append((run_id, output["hit_at_3"]))
    print(
        f"{n}/{len(tests)} segment={test['segment']} "
        f"Hit@3={output['hit_at_3']}",
        flush=True,
    )

ls.flush()
for run_id, hit in completed:
    ls.create_feedback(
        run_id=run_id,
        key="hit_at_3",
        score=hit,
        comment="Automatic evaluation against labeled relevant document IDs.",
    )

ls.flush()
print("\nFinished. Open LangSmith project: telco-churn-topic11")
