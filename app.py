import streamlit as st
import tempfile
import os
from src.rag_pipeline import RAGPipeline
from src.text_splitter import SemanticParentChildSplitter


# ─────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────

st.set_page_config(
    page_title="RAG Assistant Pro",
    page_icon="🤖",
    layout="wide"
)


# ─────────────────────────────────────────
# Initialisation du pipeline
# ─────────────────────────────────────────

@st.cache_resource
def init_pipeline():
    return RAGPipeline(top_k=3)


pipeline = init_pipeline()


# ─────────────────────────────────────────
# Fonction utilitaire — traitement des fichiers
# ─────────────────────────────────────────

def process_files(files, add_to_existing: bool):
    if not files:
        st.error("⚠️ Sélectionne des fichiers d'abord !")
        return

    with st.spinner("⏳ Indexation en cours..."):
        try:
            temp_dir = tempfile.gettempdir()
            final_paths = []

            for f in files:
                path = os.path.join(temp_dir, f.name)
                with open(path, "wb") as buffer:
                    buffer.write(f.read())
                final_paths.append(path)

            if not add_to_existing:
                pipeline.reset_index()

            pipeline.index_documents(final_paths)

            for path in final_paths:
                if os.path.exists(path):
                    os.remove(path)

            new_names = [f.name for f in files]
            if add_to_existing and "indexed_files" in st.session_state:
                st.session_state.indexed_files = list(
                    set(st.session_state.indexed_files + new_names)
                )
            else:
                st.session_state.indexed_files = new_names
                st.session_state.messages = []

            st.success(f"✅ {len(files)} fichier(s) indexé(s) !")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")


# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────

with st.sidebar:

    # ── Section 1 : Documents ──────────────
    st.header("📁 Documents")

    uploaded_files = st.file_uploader(
        label="Charge tes documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Formats supportés : PDF, TXT, DOCX"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "📥 Indexer les documents",
            use_container_width=True
        ):
            process_files(uploaded_files, add_to_existing=False)
    with col2:
        if st.button(
            "➕ Ajouter",
            use_container_width=True
        ):
            process_files(uploaded_files, add_to_existing=True)

    # Fichiers indexés
    if "indexed_files" in st.session_state:
        st.divider()
        st.subheader("📄 Fichiers indexés")
        for filename in st.session_state.indexed_files:
            st.caption(f"• {filename}")

    # Bouton reset
    st.divider()
    if st.button(
        "🗑️ Réinitialiser tout",
        use_container_width=True
    ):
        pipeline.reset_index()
        if "indexed_files" in st.session_state:
            del st.session_state.indexed_files
        if "messages" in st.session_state:
            del st.session_state.messages
        st.success("✅ Index réinitialisé")
        st.rerun()

    # ── Section 2 : Paramètres ─────────────
    # Placée en bas de la sidebar
    st.divider()
    st.header("⚙️ Paramètres avancés")

    with st.expander("🔧 Modifier les paramètres", expanded=False):

        # Top K
        st.markdown("**Nombre de sources (Top K)**")
        st.caption(
            "Nombre de passages récupérés pour répondre à chaque question."
        )
        new_top_k = st.slider(
            label="Top K",
            min_value=1,
            max_value=15,
            value=pipeline.top_k,
            label_visibility="collapsed"
        )
        if new_top_k != pipeline.top_k:
            pipeline.top_k = new_top_k
            if pipeline.is_ready():
                pipeline.retriever.set_top_k(new_top_k)
            st.success(f"✅ Top K mis à jour : {new_top_k}")

        st.divider()

        # Taille des chunks
        st.markdown("**Taille des chunks enfants (child_size)**")
        st.caption(
            "Taille des chunks de recherche. \n"
            "Le contexte fourni au LLM sera 5x plus grand (parent). \n"
            "⚠️ Nécessite une réindexation."
        )
        new_chunk_size = st.slider(
            label="Chunk size",
            min_value=100,
            max_value=1000,
            value=pipeline.splitter.child_size,
            step=50,
            label_visibility="collapsed"
        )
        if new_chunk_size != pipeline.splitter.child_size:
            pipeline.splitter = SemanticParentChildSplitter(
                embeddings=pipeline.embedding_model.get_embeddings(),
                child_size=new_chunk_size,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=80.0
            )
            st.warning(
                "⚠️ Taille modifiée — "
                f"Child : {new_chunk_size} / Parent : {5 * new_chunk_size}. "
                "Réindexe tes documents pour appliquer."
            )
        
        st.divider()

        # ── HyDE ──
        st.markdown("**HyDE — Hypothetical Document Embedding**")
        st.caption(
            "Génère une réponse hypothétique avant de chercher. "
            "Améliore le Context Recall."
        )
        hyde_enabled = st.toggle(
            label="Activer HyDE",
            value=pipeline.retriever.use_hyde
            if pipeline.is_ready() else True
        )
        if pipeline.is_ready() and hyde_enabled != pipeline.retriever.use_hyde:
            pipeline.retriever.toggle_hyde(hyde_enabled)
            st.success(
                f"✅ HyDE {'activé' if hyde_enabled else 'désactivé'}"
            )

        st.divider()

        # ── Multi-query ──
        st.markdown("**Multi-Query Retrieval**")
        st.caption(
            "Génère plusieurs reformulations de ta question. "
            "Améliore la couverture des chunks."
        )
        multi_query_enabled = st.toggle(
            label="Activer Multi-query",
            value=pipeline.retriever.use_multi_query
            if pipeline.is_ready() else True
        )
        if pipeline.is_ready() and (
            multi_query_enabled != pipeline.retriever.use_multi_query
        ):
            pipeline.retriever.toggle_multi_query(multi_query_enabled)
            st.success(
                f"✅ Multi-query "
                f"{'activé' if multi_query_enabled else 'désactivé'}"
            )

        if multi_query_enabled:
            st.markdown("**Nombre de reformulations**")
            st.caption("Plus de reformulations = meilleure couverture mais plus lent.")
            new_n_queries = st.slider(
                label="N queries",
                min_value=1,
                max_value=5,
                value=pipeline.retriever.multi_query.n_queries
                if pipeline.is_ready() and pipeline.retriever.multi_query
                else 3,
                label_visibility="collapsed"
            )
            if (
                pipeline.is_ready()
                and pipeline.retriever.multi_query
                and new_n_queries != pipeline.retriever.multi_query.n_queries
            ):
                pipeline.retriever.multi_query.set_n_queries(new_n_queries)
                st.success(f"✅ N queries mis à jour : {new_n_queries}")


# ─────────────────────────────────────────
# Zone principale — Chat
# ─────────────────────────────────────────

st.title("🤖 RAG Assistant Pro")
st.markdown(
    "Pose des questions sur tes documents **PDF**, **TXT** et **DOCX**."
)

# Initialise l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affiche l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources consultées (Preuve sémantique)"):
                for source in message["sources"]:
                    st.markdown(
                        f"📄 **{source['source']}** "
                        f"— Page {source['page']} "
                        f"— Score : `{source['score']}`"
                    )

# Champ de question
if question := st.chat_input("Pose ta question ici..."):

    if not pipeline.is_ready():
        st.warning(
            "⚠️ Charge et indexe d'abord des documents "
            "dans le panneau de gauche."
        )
    else:
        # Message utilisateur
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        with st.chat_message("user"):
            st.markdown(question)

        # Réponse assistant
        with st.chat_message("assistant"):
            with st.spinner("🔍 Recherche en cours..."):
                try:
                    # Prépare l'historique
                    chat_history = [
                        (msg["role"], msg["content"])
                        for msg in st.session_state.messages[-6:]
                    ]

                    result = pipeline.query(
                        question=question,
                        chat_history=chat_history
                    )

                    st.markdown(result["answer"])

                    # Badges des techniques actives
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if pipeline.retriever.use_hyde:
                            st.success("💭 HyDE activé")
                    with col2:
                        if pipeline.retriever.use_multi_query:
                            st.info("🔀 Multi-query activé")
                    with col3:
                        st.warning(f"🎯 Top K : {pipeline.top_k}")

                    # Question reformulée
                    if result.get("condensed_question") and \
                    result["condensed_question"] != question:
                        with st.expander("🔄 Question reformulée"):
                            st.caption(
                                f"*{result['condensed_question']}*"
                            )

                    # Sources
                    with st.expander(
                        "📚 Sources consultées (Preuve sémantique)"
                    ):
                        for source in result["sources"]:
                            st.markdown(
                                f"📄 **{source['source']}** "
                                f"— Page {source['page']} "
                                f"— Score : `{source['score']}`"
                            )

                    # Sauvegarde
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"]
                    })

                except Exception as e:
                    st.error(f"❌ Erreur : {str(e)}")

# Message d'accueil
if not st.session_state.messages:
    st.info(
        "👈 Commence par charger tes documents "
        "dans le panneau de gauche, "
        "puis pose tes questions ici !"
    )
