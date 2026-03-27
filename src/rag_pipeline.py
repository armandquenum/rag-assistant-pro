import os
import shutil
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.document_loader import DocumentLoader
from src.text_splitter import SemanticParentChildSplitter
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever

load_dotenv()


class RAGPipeline:
    """
    Orchestrateur principal du pipeline RAG Avancé.

    Flux complet :
    Documents
        → SemanticChunker (ruptures sémantiques)
        → ParentChildSplitter (Parents + Enfants)
        → FAISS (Enfants indexés)
        → Retriever (Enfants → Parents)
        → LLM Groq (génération)
        → Réponse + Sources
    """

    DEFAULT_MODEL = "llama-3.1-8b-instant"

    PROMPT_TEMPLATE = """
    Tu es un assistant expert qui répond aux questions
    en te basant UNIQUEMENT sur le contexte fourni.

    Règles importantes :
    - Réponds uniquement avec les informations du contexte
    - Si la réponse n'est pas dans le contexte, dis-le clairement
    - Cite toujours tes sources (Nom du fichier et page)
    - Réponds en français
    - Ne divulgue jamais qui tu es

    Contexte :
    {context}

    Question : {question}

    Réponse :
    """

    # Ajoute ce prompt à côté de PROMPT_TEMPLATE
    CONDENSE_QUESTION_TEMPLATE = """
    Étant donné l'historique de conversation suivant et une question de suivi,
    reformule la question de suivi en une question autonome et complète
    qui peut être comprise sans l'historique.

    Si la question est déjà autonome, retourne-la telle quelle.
    Ne réponds pas à la question, reformule-la seulement.

    Historique de conversation :
    {chat_history}

    Question de suivi : {question}

    Question reformulée :
    """

    def __init__(self, model_name: str = None, top_k: int = 3):
        """
        Args:
            model_name: modèle LLM Groq
            top_k: nombre de parents retournés au LLM
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.top_k = top_k

        # Composants
        self.loader = DocumentLoader()
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(self.embedding_model)
        self.retriever = None

        # Splitter sémantique — utilise le même modèle d'embedding
        self.splitter = SemanticParentChildSplitter(
            embeddings=self.embedding_model.get_embeddings(),
            child_size=300,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=80.0
        )

        # LLM
        self._init_llm()
        self._init_condense_prompt()

        # Prompt
        self.prompt = ChatPromptTemplate.from_template(
            self.PROMPT_TEMPLATE
        )

        print(
            f"✅ RAGPipeline Advanced initialisé\n"
            f"   → Modèle LLM : {self.model_name}\n"
            f"   → Top K : {self.top_k}\n"
            f"   → Découpage : Sémantique + Parent-Child"
        )

    def _init_llm(self):
        """Initialisation sécurisée du LLM Groq."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY manquante dans le fichier .env"
            )

        self.llm = ChatGroq(
            model=self.model_name,
            temperature=0,
            api_key=api_key
        )
        print(f"✅ LLM Groq initialisé : {self.model_name}")


    def _init_condense_prompt(self):
        """
        Initialise le prompt de reformulation de question.
        Appelé une fois dans __init__.
        """
        self.condense_prompt = ChatPromptTemplate.from_template(
            self.CONDENSE_QUESTION_TEMPLATE
        )

    def _condense_question(
        self,
        question: str,
        chat_history: list
    ) -> str:
        """
        Reformule la question en tenant compte de l'historique.
        Si pas d'historique, retourne la question telle quelle.

        Args:
            question: question actuelle de l'utilisateur
            chat_history: liste de tuples (role, message)

        Returns:
            Question reformulée et autonome
        """
        # Pas d'historique → pas besoin de reformuler
        if not chat_history:
            return question

        # Formate l'historique en texte lisible
        history_text = ""
        for role, message in chat_history:
            prefix = "Humain" if role == "user" else "Assistant"
            history_text += f"{prefix} : {message}\n"

        # Reformule la question
        prompt_value = self.condense_prompt.invoke({
            "chat_history": history_text,
            "question": question
        })

        condensed = self.llm.invoke(prompt_value)
        condensed_question = condensed.content.strip()

        print(f"\n🔄 Question reformulée : '{condensed_question}'")
        return condensed_question

    def index_documents(
        self,
        file_paths: list,
        add_to_existing: bool = False
    ):
        """
        Charge, découpe et indexe les documents.

        Args:
            file_paths: liste de chemins vers les fichiers
            add_to_existing: True = ajoute, False = recrée
        """
        print(f"\n📚 Indexation de {len(file_paths)} fichier(s)...")

        # Étape 1 — Chargement
        documents = self.loader.load_multiple_files(file_paths)
        if not documents:
            raise ValueError("Aucun document chargé.")

        # Étape 2 — Découpage sémantique + Parent-Child
        all_parents, all_children = self.splitter.split(documents)

        # Étape 3 — Indexation
        if add_to_existing and self.vector_store.exists():
            if self.vector_store.vector_store is None:
                self.vector_store.load()
            self.vector_store.add_incremental(
                parents=all_parents,
                children=all_children
            )
        else:
            self.vector_store.create_from_chunks(
                parents=all_parents,
                children=all_children
            )

        # Étape 4 — Sauvegarde sécurisée
        self.vector_store.save()

        # Étape 5 — Initialise le retriever
        self.retriever = Retriever(self.vector_store, self.top_k)

        print(f"\n✅ Indexation terminée — RAG prêt !")

    def load_existing_index(self):
        """Charge un index existant sans réindexer."""
        if not self.vector_store.exists():
            raise FileNotFoundError(
                "Aucun index existant. "
                "Appelez d'abord index_documents()."
            )

        self.vector_store.load()
        self.retriever = Retriever(self.vector_store, self.top_k)
        print(f"✅ Index chargé avec succès.")

    def query(
        self,
        question: str,
        chat_history: list = None
    ) -> dict:
        """
        Pose une question au RAG avec historique conversationnel.

        Args:
            question: question de l'utilisateur
            chat_history: liste de tuples (role, message)
                        ex: [("user", "Bonjour"), ("assistant", "Bonjour !")]

        Returns:
            Dict : answer, sources, context, condensed_question
        """
        if not self.retriever:
            raise ValueError(
                "RAG non initialisé. "
                "Appelez index_documents()."
            )

        if not question.strip():
            raise ValueError("La question est vide.")

        chat_history = chat_history or []

        print(f"\n💬 Question originale : {question}")

        # Étape 1 — Reformule la question avec l'historique
        condensed_question = self._condense_question(
            question,
            chat_history
        )

        # Étape 2 — Récupère les chunks pertinents
        # sur la question reformulée
        retrieved_data = self.retriever.retrieve_and_format(
            condensed_question
        )

        # Étape 3 — Construit le prompt avec historique
        prompt_value = self.prompt.invoke({
            "context": retrieved_data["context"],
            "question": condensed_question
        })

        # Étape 4 — Génère la réponse
        print(f"⏳ Génération via {self.model_name}...")
        response = self.llm.invoke(prompt_value)

        return {
            "answer": response.content,
            "sources": retrieved_data["sources"],
            "context": retrieved_data["context"],
            "chunks": retrieved_data["chunks"],      # ← obligatoire
            "condensed_question": condensed_question
        }

    def reset_index(self):
        """Supprime complètement l'index."""
        save_path = self.vector_store.DEFAULT_SAVE_PATH
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
            self.vector_store.vector_store = None
            self.retriever = None
            print(f"✅ Index supprimé.")
        else:
            print(f"ℹ️ Aucun index à supprimer.")

    def is_ready(self) -> bool:
        """Vérifie si le RAG est prêt."""
        return self.retriever is not None