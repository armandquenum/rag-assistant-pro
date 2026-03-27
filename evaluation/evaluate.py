"""
Évaluation du RAG avec RAGAS.

Usage :
    python -m evaluation.evaluate
"""

import os
import json
from datetime import datetime
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from src.rag_pipeline import RAGPipeline
from evaluation.test_dataset import (
    TEST_DATASET,
    get_unique_documents,
    get_questions,
    get_ground_truths,
    preview_dataset
)


def run_evaluation():
    """
    Lance l'évaluation complète du RAG.
    """

    print("="*60)
    print("🧪 ÉVALUATION RAGAS — RAG Assistant Pro")
    print("="*60)

    # ── Étape 1 : Initialisation ──
    print("\n⏳ Initialisation du pipeline...")
    pipeline = RAGPipeline(top_k=3)

    # ── Étape 2 : Indexation ──
    
    preview_dataset(TEST_DATASET, n=3) # Aperçu du dataset

    documents = get_unique_documents(TEST_DATASET)
    print(f"\n📚 Indexation de {len(documents)} document(s)...")

    missing = [d for d in documents if not os.path.exists(d)]
    if missing:
        print(f"❌ Documents manquants :")
        for m in missing:
            print(f"   → {m}")
        return

    pipeline.reset_index()
    pipeline.index_documents(documents)

    # ── Étape 3 : Génération des réponses ──
    print(
        f"\n💬 Génération des réponses "
        f"sur {len(TEST_DATASET)} questions..."
    )

    questions = []
    answers = []
    contexts = []
    ground_truths = []
    failed = []

    for i, item in enumerate(TEST_DATASET, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n[{i}/{len(TEST_DATASET)}] {question}")

        try:
            result = pipeline.query(question)

            # Vérifie que chunks est présent
            if "chunks" not in result:
                raise KeyError(
                    "'chunks' absent du résultat. "
                    "Vérifie que query() retourne bien 'chunks'."
                )

            chunks = result["chunks"]

            # Vérifie que chunks n'est pas vide
            if not chunks:
                raise ValueError("Aucun chunk retourné pour cette question.")

            questions.append(question)
            answers.append(result["answer"])
            contexts.append(chunks)
            ground_truths.append(ground_truth)

            print(f"   ✅ Réponse générée — {len(chunks)} chunk(s)")

        except Exception as e:
            print(f"   ❌ Question ignorée : {e}")
            failed.append({"question": question, "error": str(e)})

    # Vérifie qu'on a assez de résultats
    if not questions:
        print("\n❌ Aucune question n'a produit de résultat.")
        print("   → Vérifie que query() retourne bien 'chunks'")
        return

    print(f"\n✅ {len(questions)} question(s) réussies")
    if failed:
        print(f"⚠️ {len(failed)} question(s) ignorées")

    # ── Étape 4 : Configuration RAGAS avec Gemini ──
    print(f"\n⚙️ Configuration RAGAS avec Google Gemini...")

    
    google_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
        disable_streaming=True
    )
    
    ragas_llm = LangchainLLMWrapper(google_llm)

    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    print("✅ RAGAS configuré avec Gemini (Juge) + HuggingFace (Embeddings)")

    # ── Étape 5 : Calcul des métriques RAGAS ──
    print(f"\n📊 Calcul des métriques RAGAS...")

    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    results = evaluate(
        dataset=eval_dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextRecall(),
            ContextPrecision()
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(
            max_workers=1,      # ← 1 seul appel à la fois
            timeout=120,        # ← 2 minutes par appel
            max_retries=3       # ← 3 tentatives si timeout
        )
    )

    # ── Étape 6 : Affichage et sauvegarde ──
    print("\n" + "="*60)
    print("📊 RÉSULTATS DE L'ÉVALUATION")
    print("="*60)

    import math

    def _extract_score(value) -> float:
        """
        Extrait un score float depuis un résultat RAGAS.
        Gère les cas : float, int, liste, None, NaN.
        """
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            if math.isnan(float(value)):
                print("   ⚠️ Score NaN détecté — probablement un timeout")
                return 0.0
            return float(value)
        if isinstance(value, list):
            valid = [
                float(v) for v in value
                if v is not None and not math.isnan(float(v))
            ]
            if not valid:
                print("   ⚠️ Tous les scores sont NaN — timeout Groq")
                return 0.0
            return sum(valid) / len(valid)
        return float(value)

    scores = {
        "faithfulness": round(
            _extract_score(results["faithfulness"]), 4
        ),
        "answer_relevancy": round(
            _extract_score(results["answer_relevancy"]), 4
        ),
        "context_recall": round(
            _extract_score(results["context_recall"]), 4
        ),
        "context_precision": round(
            _extract_score(results["context_precision"]), 4
        ),
    }


    for metric, score in scores.items():
        emoji = (
            "✅" if score >= 0.7
            else "⚠️" if score >= 0.5
            else "❌"
        )
        print(f"   {emoji} {metric:<25} : {score:.4f}")

    avg_score = sum(scores.values()) / len(scores)
    print(f"\n   📈 Score global moyen    : {avg_score:.4f}")

    print("\n" + "="*60)
    print("💡 INTERPRÉTATION")
    print("="*60)
    _interpret_scores(scores)

    # Sauvegarde
    os.makedirs("evaluation/results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"evaluation/results/ragas_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "scores": scores,
                "average": round(avg_score, 4),
                "num_questions": len(questions),
                "num_failed": len(failed),
                "model": pipeline.model_name,
                "top_k": pipeline.top_k,
                "failed_questions": failed
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"\n💾 Résultats sauvegardés : {output_path}")
    return scores


def _interpret_scores(scores: dict):
    """Interprète les scores et donne des conseils."""

    f = scores["faithfulness"]
    if f >= 0.8:
        print("   ✅ Faithfulness — le RAG n'hallucine pas")
    elif f >= 0.5:
        print(
            "   ⚠️ Faithfulness moyenne\n"
            "      → Augmente top_k ou améliore le prompt"
        )
    else:
        print(
            "   ❌ Faithfulness faible — beaucoup d'hallucinations\n"
            "      → Vérifie le prompt et le contexte fourni"
        )

    ar = scores["answer_relevancy"]
    if ar >= 0.8:
        print("   ✅ Answer Relevancy — réponses bien ciblées")
    elif ar >= 0.5:
        print(
            "   ⚠️ Answer Relevancy moyenne\n"
            "      → Reformule le PROMPT_TEMPLATE"
        )
    else:
        print(
            "   ❌ Answer Relevancy faible\n"
            "      → Vérifie la qualité des chunks retournés"
        )

    cr = scores["context_recall"]
    if cr >= 0.8:
        print("   ✅ Context Recall — bons chunks trouvés")
    elif cr >= 0.5:
        print(
            "   ⚠️ Context Recall partiel\n"
            "      → Augmente top_k ou réduis child_size"
        )
    else:
        print(
            "   ❌ Context Recall faible\n"
            "      → Vérifie le découpage sémantique"
        )

    cp = scores["context_precision"]
    if cp >= 0.8:
        print("   ✅ Context Precision — peu de bruit")
    elif cp >= 0.5:
        print(
            "   ⚠️ Context Precision moyenne\n"
            "      → Réduis top_k ou augmente le seuil sémantique"
        )
    else:
        print(
            "   ❌ Context Precision faible\n"
            "      → Améliore le reranking"
        )


if __name__ == "__main__":
    run_evaluation()