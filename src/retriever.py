from src.vector_store import VectorStore
from src.reranker import Reranker
from src.hyde import HyDE
from src.multi_query import MultiQueryGenerator
from langchain_core.documents import Document


class Retriever:
    """
    Classe responsable de la recherche des documents.

    Flux avancé :
    1. Multi-query  → génère N reformulations
    2. HyDE         → génère une réponse hypothétique par reformulation
    3. FAISS        → N+1 recherches (originale + HyDE)
    4. Fusion       → déduplique les résultats
    5. Parent-Child → remonte aux Parents
    6. Reranking    → reclasse par pertinence
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 3,
        llm=None,
        use_hyde: bool = True,
        use_multi_query: bool = True,
        n_queries: int = 3
    ):
        """
        Args:
            vector_store: instance de VectorStore
            top_k: nombre de documents finals au LLM
            llm: LLM pour HyDE et Multi-query
            use_hyde: active/désactive HyDE
            use_multi_query: active/désactive Multi-query
            n_queries: nombre de reformulations Multi-query
        """
        self.vector_store_manager = vector_store
        self.store = vector_store.get_store()
        self.parents_store = vector_store.get_parents_store()
        self.top_k = top_k
        self.reranker = Reranker()

        # HyDE et Multi-query
        self.use_hyde = use_hyde and llm is not None
        self.use_multi_query = use_multi_query and llm is not None

        if llm is not None:
            self.hyde = HyDE(llm) if use_hyde else None
            self.multi_query = MultiQueryGenerator(
                llm, n_queries
            ) if use_multi_query else None
        else:
            self.hyde = None
            self.multi_query = None
            if use_hyde or use_multi_query:
                print(
                    "⚠️ HyDE et Multi-query désactivés "
                    "— LLM non fourni"
                )

        print(
            f"✅ Retriever initialisé\n"
            f"   → HyDE : {'activé' if self.use_hyde else 'désactivé'}\n"
            f"   → Multi-query : "
            f"{'activé' if self.use_multi_query else 'désactivé'}\n"
            f"   → Top K : {top_k}"
        )

    def _search_faiss(
        self,
        query: str,
        k: int
    ) -> list:
        """
        Effectue une recherche FAISS pour une query.

        Args:
            query: texte à rechercher
            k: nombre de résultats

        Returns:
            Liste de tuples (Document, score)
        """
        return self.store.similarity_search_with_score(
            query=query,
            k=k
        )

    def _fuse_results(self, all_results: list) -> list:
        """
        Fusionne et déduplique les résultats de plusieurs recherches.
        Garde le meilleur score pour chaque chunk dupliqué.

        Args:
            all_results: liste de listes de (Document, score)

        Returns:
            Liste dédupliquée de (Document, score)
        """
        seen_contents = {}

        for results in all_results:
            for doc, score in results:
                content = doc.page_content

                # Garde le meilleur score (plus bas = plus similaire)
                if content not in seen_contents:
                    seen_contents[content] = (doc, score)
                else:
                    existing_score = seen_contents[content][1]
                    if score < existing_score:
                        seen_contents[content] = (doc, score)

        fused = list(seen_contents.values())

        print(
            f"\n🔀 Fusion : {sum(len(r) for r in all_results)} "
            f"résultats → {len(fused)} uniques"
        )

        return fused

    def retrieve(self, query: str, k: int = None) -> list:
        """
        Recherche avancée avec HyDE + Multi-query + FAISS.

        Args:
            query: question de l'utilisateur
            k: nombre de candidats par recherche

        Returns:
            Liste fusionnée de (Document enfant, score)
        """
        if not query or not query.strip():
            raise ValueError("La question ne peut pas être vide.")

        search_k = k or (self.top_k * 4)
        all_results = []

        # ── Étape 1 : Génère les queries ──
        if self.use_multi_query:
            queries = self.multi_query.generate_queries(query)
        else:
            queries = [query]

        # ── Étape 2 : Pour chaque query, applique HyDE ──
        search_queries = []

        for q in queries:
            if self.use_hyde:
                hyp_doc = self.hyde.generate_hypothetical_document(q)
                search_queries.append(hyp_doc)
            else:
                search_queries.append(q)

        # Ajoute toujours la question originale
        if query not in search_queries:
            search_queries.append(query)

        print(
            f"\n🔍 {len(search_queries)} recherche(s) FAISS lancée(s)"
        )

        # ── Étape 3 : Recherches FAISS ──
        for sq in search_queries:
            results = self._search_faiss(sq, search_k)
            all_results.append(results)
            print(
                f"   → '{sq[:50]}...' "
                f"→ {len(results)} résultat(s)"
            )

        # ── Étape 4 : Fusion ──
        fused_results = self._fuse_results(all_results)

        return fused_results

    def retrieve_and_format(self, query: str) -> dict:
        """
        Pipeline complet :
        Multi-query → HyDE → FAISS → Parents → Reranking → Formatage

        Args:
            query: question de l'utilisateur

        Returns:
            Dict : chunks, sources, scores, context
        """

        # ── Étape 1 : Recherche avancée ──
        raw_results = self.retrieve(query)

        # ── Étape 2 : Remontée aux Parents ──
        unique_parents = {}
        child_scores = {}

        for child_doc, score in raw_results:
            parent_id = child_doc.metadata.get("parent_id")

            if parent_id and parent_id not in unique_parents:
                parent_doc = self.parents_store.get(parent_id)

                if parent_doc is not None:
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

        # ── Étape 3 : Reranking ──
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

            # Priorité au score reranker normalisé
            # Fallback sur score FAISS si reranker non disponible
            display_score = doc.metadata.get(
                "rerank_score",
                doc.metadata.get("child_score", 0.0)
            )

            sources.append({
                "source": doc.metadata.get("source", "inconnu"),
                "page": doc.metadata.get("page", "N/A"),
                "score": display_score
            })
            scores.append(display_score)

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
        """Formate les chunks en contexte pour le LLM."""
        context_parts = []
        for i, (chunk, source) in enumerate(
            zip(chunks, sources), 1
        ):
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
                f"— score rerank : {s['score']} "
                f"(entre 0 et 1)"
            )

    def set_top_k(self, top_k: int):
        """Met à jour le nombre de documents retournés."""
        if top_k < 1:
            raise ValueError("top_k doit être supérieur à 0.")
        self.top_k = top_k
        print(f"✅ top_k mis à jour : {self.top_k}")

    def toggle_hyde(self, enabled: bool):
        """Active ou désactive HyDE."""
        if self.hyde is None and enabled:
            print("⚠️ HyDE ne peut pas être activé — LLM non fourni")
            return
        self.use_hyde = enabled
        print(f"✅ HyDE : {'activé' if enabled else 'désactivé'}")

    def toggle_multi_query(self, enabled: bool):
        """Active ou désactive Multi-query."""
        if self.multi_query is None and enabled:
            print(
                "⚠️ Multi-query ne peut pas être activé "
                "— LLM non fourni"
            )
            return
        self.use_multi_query = enabled
        print(
            f"✅ Multi-query : "
            f"{'activé' if enabled else 'désactivé'}"
        )