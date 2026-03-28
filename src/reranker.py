from sentence_transformers import CrossEncoder
import numpy as np


class Reranker:
    """
    Classe responsable du re-classement des chunks.
    Passe d'une recherche statistique à une analyse sémantique profonde.

    Scores normalisés entre 0 et 1 via sigmoid :
    Score proche de 1 → très pertinent ✅
    Score proche de 0 → peu pertinent  ❌
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Args:
            model_name: modèle CrossEncoder HuggingFace
        """
        print(f"⏳ Chargement du Re-ranker : {model_name}...")
        self.model = CrossEncoder(model_name, device="cpu")
        print("✅ Re-ranker prêt.")

    def _sigmoid(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalise les scores bruts entre 0 et 1.

        Le CrossEncoder retourne des logits sans borne
        → sigmoid ramène tout entre 0 et 1

        Args:
            scores: scores bruts du CrossEncoder

        Returns:
            Scores normalisés entre 0 et 1
        """
        return 1 / (1 + np.exp(-scores))

    def rerank(self, query: str, documents: list) -> list:
        """
        Ré-ordonne les documents par ordre de pertinence réelle.

        Args:
            query: question de l'utilisateur
            documents: liste de documents LangChain

        Returns:
            Liste de documents triés du plus au moins pertinent
            avec scores normalisés dans les métadonnées
        """
        if not documents:
            return []

        # Paires (Question, Document) pour le CrossEncoder
        pairs = [
            [query, doc.page_content]
            for doc in documents
        ]

        # Scores bruts du CrossEncoder — logits sans borne
        raw_scores = self.model.predict(pairs)

        # Normalisation sigmoid → scores entre 0 et 1
        normalized_scores = self._sigmoid(raw_scores)

        # Tri par score décroissant
        sorted_indices = np.argsort(normalized_scores)[::-1]
        reranked_docs = [documents[i] for i in sorted_indices]

        # Stocke les scores normalisés dans les métadonnées
        for rank, idx in enumerate(sorted_indices):
            reranked_docs[rank].metadata["rerank_score"] = round(
                float(normalized_scores[idx]), 4
            )

        print(
            f"✅ Re-ranking terminé sur {len(documents)} documents.\n"
            f"   → Scores normalisés (sigmoid) : "
            f"{[round(float(s), 4) for s in normalized_scores[sorted_indices]]}"
        )

        return reranked_docs