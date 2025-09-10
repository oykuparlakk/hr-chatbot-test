import os
import time
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rag import rag_pipeline

# ENV değişkenlerini yükle
load_dotenv()
EVAL_PATH = os.getenv("EVAL_PATH", "./eval/golden_eval_question_set.jsonl")
TOP_K = int(os.getenv("TOP_K", 5))


def run_eval():
    # JSONL dosyasını oku
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_data = [json.loads(line) for line in f]

    results = []
    for i, item in enumerate(eval_data, 1):
        query = item["Question"]
        expected_doc = item.get("expected_doc")
        expected_section = item.get("expected_section")
        is_answerable = item.get("answerable", True)

        # anchor_keywords string → listeye çevir (None kontrolü ile)
        raw_keywords = item.get("anchor_keywords")
        if raw_keywords:
            expected_keywords = [
                kw.strip(" '\"") for kw in raw_keywords.split(",") if kw.strip()
            ]
        else:
            expected_keywords = []

        print(f"\n🔹 [{i}/{len(eval_data)}] Soru: {query}")

        # RAG pipeline çağır
        answer, sources, latency = rag_pipeline(query, stream=False)

        # Recall hesapla
        retrieved_docs = [s["source"] for s in sources]
        recall_hit = expected_doc and any(expected_doc in d for d in retrieved_docs)

        # Groundedness (keyword check)
        grounded_hits = [kw for kw in expected_keywords if kw.lower() in answer.lower()]
        groundedness = (
            len(grounded_hits) / max(1, len(expected_keywords))
            if expected_keywords else None
        )

        # Answerable / Unanswerable
        if not is_answerable:
            unanswerable_correct = "bilmiyorum" in answer.lower()
        else:
            unanswerable_correct = None

        # Kaydet
        results.append({
            "question": query,
            "expected_doc": expected_doc,
            "expected_section": expected_section,
            "retrieved_docs": retrieved_docs,
            "recall_hit": recall_hit,
            "groundedness": groundedness,
            "latency": latency,
            "answer": answer,
            "is_answerable": is_answerable,
            "unanswerable_correct": unanswerable_correct
        })

    # DataFrame
    df = pd.DataFrame(results)

    # Metrikler
    recall = df["recall_hit"].mean() * 100
    avg_latency = df["latency"].mean()
    p95_latency = np.percentile(df["latency"], 95)

    hallucination_rate = (
        (1 - df["groundedness"].mean()) * 100
        if df["groundedness"].notna().any() else None
    )

    if "unanswerable_correct" in df.columns:
        unanswerable_accuracy = (
            df["unanswerable_correct"].dropna().mean() * 100
            if not df["unanswerable_correct"].dropna().empty else None
        )
    else:
        unanswerable_accuracy = None

    # Özet
    print("\n==== 📊 EVAL SONUÇLARI ====")
    print(f"Recall@{TOP_K}: {recall:.2f}%")
    print(f"Ortalama Latency: {avg_latency:.2f}s")
    print(f"P95 Latency: {p95_latency:.2f}s")
    if hallucination_rate is not None:
        print(f"Hallucination Rate: {hallucination_rate:.2f}%")
    if unanswerable_accuracy is not None:
        print(f"Unanswerable Accuracy: {unanswerable_accuracy:.2f}%")

    # CSV export
    df.to_csv("./eval/eval_results.csv", index=False)


if __name__ == "__main__":
    run_eval()
 