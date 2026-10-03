import json
from pathlib import Path
from time import perf_counter
from fastembed.rerank.cross_encoder import TextCrossEncoder

docs = json.loads(Path("data/documents.json").read_text())
by_id = {d["id"]: d for d in docs}
evaluation = json.loads(Path("results/retrieval_eval.json").read_text())
rows = evaluation["queries"]

model = "BAAI/bge-reranker-base"
print("Loading:", model, flush=True)
reranker = TextCrossEncoder(model_name=model, threads=2)

def metrics(ids, relevant):
    ranks = [
        rank for rank, doc_id in enumerate(ids, 1)
        if doc_id in relevant
    ]
    return {
        "Hit@1": float(any(r <= 1 for r in ranks)),
        "Hit@3": float(any(r <= 3 for r in ranks)),
        "MRR@10": 1 / min(ranks) if ranks else 0.0,
    }

for n, row in enumerate(rows, 1):
    candidates = row["hybrid_ids"][:10]
    texts = [by_id[i]["text"] for i in candidates]
    start = perf_counter()
    scores = list(reranker.rerank(row["query"], texts, batch_size=1))
    row["rerank_ms"] = (perf_counter() - start) * 1000
    ranked = sorted(
        zip(candidates, scores), key=lambda pair: (-float(pair[1]), pair[0])
    )
    row["reranked_ids"] = [i for i, _ in ranked]
    row["reranker_scores"] = [float(s) for _, s in ranked]
    row["reranked"] = metrics(row["reranked_ids"], row["relevant_ids"])
    print(
        f"{n}/{len(rows)} top-3={row['reranked_ids'][:3]} "
        f"time={row['rerank_ms']:.0f} ms",
        flush=True,
    )

summary = {}
print(f"\n{'Method':<12} {'Hit@1':>8} {'Hit@3':>8} {'MRR@10':>8}")
for method in ("dense", "hybrid", "reranked"):
    m = {
        key: sum(r[method][key] for r in rows) / len(rows)
        for key in ("Hit@1", "Hit@3", "MRR@10")
    }
    summary[method] = m
    print(
        f"{method:<12} {m['Hit@1']:8.3f} "
        f"{m['Hit@3']:8.3f} {m['MRR@10']:8.3f}"
    )

avg_ms = sum(r["rerank_ms"] for r in rows) / len(rows)
print(f"\nMean reranking time: {avg_ms:.0f} ms/query")
Path("results/reranker_eval.json").write_text(
    json.dumps({
        "model": model,
        "summary": summary,
        "mean_rerank_ms": avg_ms,
        "queries": rows,
    }, indent=2)
)
print("Saved: results/reranker_eval.json")
