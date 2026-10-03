# Topic 11: Retrieval Systems & Knowledge Infrastructure

Telco churn knowledge retrieval with Qdrant, hybrid search, reranking,
and LangSmith tracing. Executed on an Ubuntu AWS EC2 VM using CPU.

## Dataset and indexing

- 12 synthetic documents: tariffs, retention playbooks, and complaints.
- Metadata includes document type, customer segment, and synthetic-data flag.
- Qdrant collection: `churn_kb`.
- Embeddings: `BAAI/bge-small-en-v1.5`, 384 dimensions.
- Cosine similarity; HNSW configuration: m=16, ef_construct=128.
- Payload indexes and a segment-filtered retrieval example.

## Retrieval evaluation

18 labeled queries, searched across all segments without a segment filter.
Hybrid retrieval combines dense search and BM25 using Reciprocal Rank Fusion.
The cross-encoder reranks up to 10 hybrid candidates.

| Method | Hit@1 | Hit@3 | MRR@10 |
|---|---:|---:|---:|
| Dense | 0.833 | 1.000 | 0.907 |
| Hybrid | 0.889 | 1.000 | 0.944 |
| Reranked | 0.944 | 1.000 | 0.972 |

Reranker: `BAAI/bge-reranker-base`.
Mean measured reranking time: approximately 922 ms/query on this VM.

Hybrid Hit@3 by segment:
- Consumer: 1.000, 10 queries.
- SMB: 1.000, 4 queries.
- Enterprise: 1.000, 4 queries.

These results describe a small synthetic demonstration, not a production benchmark.
The evaluation contains 18 queries rather than the 200-query benchmark discussed
in the slides. NDCG@5 is not measured. The CPU-compatible reranker used here is
bge-reranker-base rather than the slides' bge-reranker-v2-m3.

## LangSmith

Project: `telco-churn-topic11`, EU endpoint.

18 hybrid retrieval traces include embedding, Qdrant dense search,
BM25 search, and RRF fusion. Each root run includes segment metadata
and a `hit_at_3` feedback score. The Monitoring dashboard groups
feedback by Consumer, SMB, and Enterprise.

## Run

Requires Python 3.10 and Docker.

```bash
docker volume create qdrant_topic11_data
docker run -d --name qdrant-topic11 \
  -p 127.0.0.1:6333:6333 \
  -v qdrant_topic11_data:/qdrant/storage \
  qdrant/qdrant:v1.19.1

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python index_documents.py
python evaluate.py
python evaluate_reranker.py
