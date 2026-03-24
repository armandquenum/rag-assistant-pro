from sentence_transformers import CrossEncoder
import numpy as np

class Reranker:
    """
    Classe responsable du re-classement des chunks.
    Passe d'une recherche statistique à une analyse sémantique profonde.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Args:
            model_name: Modèle de re-ranking HuggingFace.
        """
        print(f"⏳ Chargement du Re-ranker : {model_name}...")
        self.model = CrossEncoder(model_name, device="cpu")
        print("✅ Re-ranker prêt.")

    def rerank(self, query: str, documents: list) -> list:
        """
        Ré-ordonne les documents par ordre de pertinence réelle.
        
        Args:
            query: La question de l'utilisateur.
            documents: Liste de documents LangChain issus de FAISS.
            
        Returns:
            Liste de documents triés du plus au moins pertinent.
        """
        if not documents:
            return []

        # Préparation des paires (Question, Document) pour le modèle
        # Format attendu : [[Q, D1], [Q, D2], ...]
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Calcul des scores de pertinence
        # Score : float unique représentant la proximité sémantique
        scores = self.model.predict(pairs)
        
        # Association des scores aux documents et tri
        # On utilise argsort pour obtenir les indices triés (ordre décroissant)
        sorted_indices = np.argsort(scores)[::-1]
        
        reranked_docs = [documents[i] for i in sorted_indices]
        
        # COMMENTAIRE TECHNIQUE :
        # Input : N paires de textes.
        # Output : Vecteur de scores de dimension (N,).
        # Complexité : O(N * L^2) où L est la longueur des séquences.
        print(f"✅ Re-ranking terminé sur {len(documents)} documents.")
        return reranked_docs