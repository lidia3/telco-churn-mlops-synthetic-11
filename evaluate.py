import json
import re
from pathlib import Path
from time import perf_counter

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

docs = json.loads(Path("data/documents.json").read_text())
by_id = {d["id"]: d for d in docs}
bm25 = BM25Okapi([tokenize(d["text"]) for d in docs])

tests = [
    ("What does the basic mobile plan cost?", [1], "Consumer"),
    ("Which plan includes unlimited calls and data?", [2], "Consumer"),
    ("What is the EU roaming allowance on Unlimited?", [2], "Consumer"),
    ("Is there a business plan for five phone lines?", [3], "SMB"),
    ("Which service has dedicated fiber and an availability SLA?", [4], "Enterprise"),
    ("A customer cannot afford the bill and wants to cancel.", [5], "Consumer"),
    ("What discount can we offer to retain a consumer?", [5], "Consumer"),
    ("A small business wants to leave after recurring outages.", [6], "SMB"),
    ("How should we handle an enterprise SLA breach?", [7], "Enterprise"),
    ("A customer wants to cancel because of weak mobile coverage.", [8], "Consumer"),
    ("Who received a refund for a duplicate September payment?", [9], "Consumer"),
    ("What happened in complaint C001?", [9], "Consumer"),
    ("Which complaint involved a faulty office router?", [10], "SMB"),
    ("What happened in complaint S001?", [10], "SMB"),
    ("Which enterprise incident caused four hours of downtime?", [11], "Enterprise"),
    ("What happened in complaint E001?", [11], "Enterprise"),
    ("Why was mobile internet slow in the evenings?", [12], "Consumer"),
    ("What happened in complaint C002?", [12], "Consumer"),
]

Path("data/eval_queries.json").write_text(
    json.dumps([
        {"query": q, "relevant_ids": ids, "segment": seg}
        for q, ids, seg in tests
    ], indent=2)
)

client = QdrantClient(url="http://localhost:6333")
embedder = TextEmbedding("BAAI/bge-small-en-v1.5", threads=2)

def rrf(*rankings):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (60 + rank)
    return sorted(scores, key=lambda i: (-scores[i], i))

def metrics(ranking, relevant):
    ranks = [
        rank for rank, doc_id in enumerate(ranking, start=1)
        if doc_id in relevant
    ]
    return {
        "Hit@1": float(any(r <= 1 for r in ranks)),
        "Hit@3": float(any(r <= 3 for r in ranks)),
        "MRR@10": 1 / min(ranks) if ranks else 0.0,
    }

rows = []
for query, relevant, segment in tests:
    vector = next(embedder.query_embed(query)).tolist()
    start = perf_counter()
    hits = client.query_points(
        collection_name="churn_kb",
        query=vector,
        limit=10,
        with_payload=True,
        search_params=models.SearchParams(hnsw_ef=64),
    ).points
    dense = [int(h.id) for h in hits]

    scores = bm25.get_scores(tokenize(query))
    sparse = [
        docs[i]["id"]
        for i in sorted(
            range(len(docs)), key=lambda i: (-scores[i], docs[i]["id"])
        )
        if scores[i] > 0
    ][:10]
    hybrid = rrf(dense, sparse)[:10]

    rows.append({
        "query": query,
        "segment": segment,
        "relevant_ids": relevant,
        "dense_ids": dense,
        "hybrid_ids": hybrid,
        "dense": metrics(dense, relevant),
        "hybrid": metrics(hybrid, relevant),
        "search_and_fusion_ms": (perf_counter() - start) * 1000,
    })

print(f"\nEvaluation queries: {len(rows)}")
print("Search across all segments; no segment filter.")
print(f"{'Method':<12} {'Hit@1':>8} {'Hit@3':>8} {'MRR@10':>8}")
summary = {}
for method in ("dense", "hybrid"):
    summary[method] = {
        key: sum(r[method][key] for r in rows) / len(rows)
        for key in ("Hit@1", "Hit@3", "MRR@10")
    }
    m = summary[method]
    print(
        f"{method:<12} {m['Hit@1']:8.3f} "
        f"{m['Hit@3']:8.3f} {m['MRR@10']:8.3f}"
    )

print("\nHybrid Hit@3 by segment:")
for segment in ("Consumer", "SMB", "Enterprise"):
    subset = [r for r in rows if r["segment"] == segment]
    score = sum(r["hybrid"]["Hit@3"] for r in subset) / len(subset)
    print(f"{segment}: {score:.3f} ({len(subset)} queries)")

Path("results").mkdir(exist_ok=True)
Path("results/retrieval_eval.json").write_text(
    json.dumps({"summary": summary, "queries": rows}, indent=2)
)
print("\nSaved: results/retrieval_eval.json")
