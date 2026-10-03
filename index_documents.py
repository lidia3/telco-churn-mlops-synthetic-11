import json
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

COLLECTION = "churn_kb"
MODEL = "BAAI/bge-small-en-v1.5"

records = [
    ("tariff", "Consumer", "Consumer Basic costs 20 EUR monthly and includes 10 GB mobile data. Extra data costs 2 EUR per GB."),
    ("tariff", "Consumer", "Consumer Unlimited costs 35 EUR monthly with unlimited domestic calls and mobile data. EU roaming includes 15 GB."),
    ("tariff", "SMB", "SMB Team costs 100 EUR monthly for five mobile lines with shared 100 GB data and business support."),
    ("tariff", "Enterprise", "Enterprise Connect offers dedicated fiber, a 99.9 percent availability SLA and a dedicated account manager. Pricing is negotiated."),
    ("playbook", "Consumer", "When a consumer wants to cancel because the bill is too high, review usage and offer a cheaper plan. A retention discount of 10 percent for three months requires approval."),
    ("playbook", "SMB", "For an SMB customer considering cancellation after repeated outages, escalate to business support, review service credits and arrange a technical investigation."),
    ("playbook", "Enterprise", "If an enterprise customer threatens to leave after an SLA breach, notify the account manager, prepare an incident report and calculate contractual service credits."),
    ("playbook", "Consumer", "When a consumer plans to leave because of poor mobile coverage, check the coverage map, investigate the location and offer Wi-Fi calling if supported."),
    ("complaint", "Consumer", "Complaint C001: a consumer was charged twice for the September invoice. Billing confirmed a duplicate payment and issued a refund."),
    ("complaint", "SMB", "Complaint S001: an SMB office experienced three internet outages in one week. A faulty router was replaced and a service credit was granted."),
    ("complaint", "Enterprise", "Complaint E001: enterprise fiber service was unavailable for four hours. The account manager coordinated restoration and an SLA compensation review."),
    ("complaint", "Consumer", "Complaint C002: a consumer reported slow mobile internet during evenings. Network support identified congestion and opened a capacity upgrade ticket."),
]

docs = [
    {
        "id": i,
        "type": kind,
        "segment": segment,
        "text": text,
        "date": "2026-10-01",
        "synthetic": True,
    }
    for i, (kind, segment, text) in enumerate(records, start=1)
]

Path("data").mkdir(exist_ok=True)
Path("data/documents.json").write_text(
    json.dumps(docs, indent=2), encoding="utf-8"
)

print("Loading embedding model...", flush=True)
embedder = TextEmbedding(model_name=MODEL, threads=2)
vectors = list(embedder.passage_embed([d["text"] for d in docs]))
dimension = len(vectors[0])

client = QdrantClient(url="http://localhost:6333")

if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=dimension,
            distance=models.Distance.COSINE,
        ),
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
    )

for field in ("type", "segment"):
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name=field,
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True,
    )

client.upsert(
    collection_name=COLLECTION,
    points=[
        models.PointStruct(id=d["id"], vector=v.tolist(), payload=d)
        for d, v in zip(docs, vectors)
    ],
    wait=True,
)

print("Collection:", COLLECTION)
print("Embedding model:", MODEL)
print("Vector dimension:", dimension)
print("Indexed documents:", client.count(COLLECTION, exact=True).count)

query = "How should we handle an SMB customer leaving after repeated outages?"
query_vector = next(embedder.query_embed(query))

hits = client.query_points(
    collection_name=COLLECTION,
    query=query_vector.tolist(),
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="segment",
                match=models.MatchValue(value="SMB"),
            )
        ]
    ),
    search_params=models.SearchParams(hnsw_ef=64),
    limit=3,
    with_payload=True,
).points

print("\nQuery:", query)
print("Filter: segment=SMB")
for rank, hit in enumerate(hits, start=1):
    print(
        f"{rank}. id={hit.id} score={hit.score:.4f} "
        f"type={hit.payload['type']} segment={hit.payload['segment']}"
    )
    print("  ", hit.payload["text"])
