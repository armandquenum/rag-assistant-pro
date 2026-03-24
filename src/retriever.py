from src.vector_store import VectorStore
from src.reranker import Reranker
from langchain_core.documents import Document


class Retriever:
    """
    Classe responsable de la recherche des documents.

    Flux Parent-Child + Reranking :
    1. Recherche initiale sur les chunks Enfants dans FAISS
    2. Remontée aux Parents via parent_id
    3. Reranking sémantique sur les Parents
    4. Retour des meilleurs Parents au LLM
    """

    def __init__(self, vector_store: VectorStore, top_k: int = 3):
        """
        Args:
            vector_store: instance de VectorStore
            top_k: nombre de documents finals à présenter au LLM
        """
        self.vector_store_manager = vector_store
        self.store = vector_store.get_store()
        self.parents_store = vector_store.get_parents_store()
        self.top_k = top_k
        self.reranker = Reranker()

    def retrieve(self, query: str, k: int = None) -> list:
        """
        Cherche les chunks Enfants les plus similaires dans FAISS.

        Args:
            query: question de l'utilisateur
            k: nombre de candidats à extraire
               Par défaut : top_k * 4 pour avoir assez de candidats
               pour le reranking

        Returns:
            Liste de tuples (Document enfant, score)
        """
        if not query or not query.strip():
            raise ValueError("La question ne peut pas être vide.")

        # On ratisse large pour le reranking
        search_k = k or (self.top_k * 4)

        print(f"\n🔍 Recherche FAISS (Enfants) : '{query}'")
        print(f"   → {search_k} candidats enfants recherchés")

        results = self.store.similarity_search_with_score(
            query=query,
            k=search_k
        )

        print(f"   → {len(results)} enfant(s) trouvé(s)")
        return results

    def retrieve_and_format(self, query: str) -> dict:
        """
        Pipeline complet :
        Enfants FAISS → Parents → Reranking → Formatage

        Args:
            query: question de l'utilisateur

        Returns:
            Dict : chunks, sources, scores, context
        """

        # ── Étape 1 : Recherche des Enfants ──
        raw_results = self.retrieve(query)

        # ── Étape 2 : Remontée aux Parents ──
        # Dictionnaire pour éviter les doublons
        # (plusieurs enfants peuvent pointer vers le même parent)
        unique_parents = {}
        child_scores = {}

        for child_doc, score in raw_results:
            parent_id = child_doc.metadata.get("parent_id")

            if parent_id and parent_id not in unique_parents:

                # Récupère le parent depuis le dictionnaire
                parent_doc = self.parents_store.get(parent_id)

                if parent_doc is not None:
                    # Crée un Document Parent avec métadonnées complètes
                    enriched_parent = Document(
                        page_content=parent_doc.page_content,
                        metadata={
                            **child_doc.metadata,
                            **parent_doc.metadata,
                            "child_score": round(float(score), 4)
                        }
                    )
                    unique_parents[parent_id] = enriched_parent
                    child_scores[parent_id] = round(float(score), 4)

        parents_to_rerank = list(unique_parents.values())

        print(
            f"\n👥 Parents uniques récupérés : "
            f"{len(parents_to_rerank)}"
        )

        if not parents_to_rerank:
            return {
                "chunks": [],
                "sources": [],
                "scores": [],
                "context": "Aucun document pertinent trouvé."
            }

        # ── Étape 3 : Reranking sémantique sur les Parents ──
        reranked_parents = self.reranker.rerank(
            query,
            parents_to_rerank
        )

        # ── Étape 4 : Sélection finale ──
        final_selection = reranked_parents[:self.top_k]

        chunks = []
        sources = []
        scores = []

        for doc in final_selection:
            chunks.append(doc.page_content)
            sources.append({
                "source": doc.metadata.get("source", "inconnu"),
                "page": doc.metadata.get("page", "N/A"),
                "score": doc.metadata.get("child_score", 0.0)
            })
            scores.append(doc.metadata.get("child_score", 0.0))

        # Formate le contexte pour le LLM
        context = self._format_context(chunks, sources)
        self._print_summary(sources)

        return {
            "chunks": chunks,
            "sources": sources,
            "scores": scores,
            "context": context
        }

    def _format_context(
        self,
        chunks: list,
        sources: list
    ) -> str:
        """
        Formate les chunks Parents en contexte lisible pour le LLM.

        Args:
            chunks: textes des parents
            sources: métadonnées des parents

        Returns:
            Contexte formaté
        """
        context_parts = []

        for i, (chunk, source) in enumerate(zip(chunks, sources), 1):
            context_parts.append(
                f"[Document {i} — "
                f"{source['source']} "
                f"p. {source['page']}]\n{chunk}"
            )

        return "\n\n".join(context_parts)

    def _print_summary(self, sources: list):
        """Affiche les sources sélectionnées après reranking."""
        print("\n📋 Contexte final après reranking :")
        for i, s in enumerate(sources, 1):
            print(
                f"   {i}. {s['source']} "
                f"— page {s['page']} "
                f"— score : {s['score']}"
            )

    def set_top_k(self, top_k: int):
        """
        Met à jour le nombre de documents retournés.

        Args:
            top_k: nouveau nombre de documents
        """
        if top_k < 1:
            raise ValueError("top_k doit être supérieur à 0.")
        self.top_k = top_k
        print(f"✅ top_k mis à jour : {self.top_k}")

