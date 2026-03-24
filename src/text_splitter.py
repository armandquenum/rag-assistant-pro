from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import uuid


class SemanticParentChildSplitter:
    """
    Découpage en deux étapes :
    
    Étape 1 — SemanticChunker :
        Découpe le texte aux ruptures sémantiques.
        Chaque chunk = une idée cohérente.
    
    Étape 2 — ParentChildSplitter :
        Chaque chunk sémantique devient un Parent.
        Chaque Parent est redécoupé en petits Enfants.
        Les Enfants sont indexés dans FAISS.
        Les Parents sont stockés pour le contexte LLM.
    
    Avantage :
        Les chunks respectent le sens du texte
        ET bénéficient de la recherche précise Parent-Child.
    """

    def __init__(
        self,
        embeddings,
        child_size: int = 300,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95.0
    ):
        """
        Args:
            embeddings: modèle d'embedding pour la détection sémantique
            child_size: taille des chunks enfants en caractères
            breakpoint_threshold_type: méthode de détection des ruptures
                - "percentile" : coupe si dissimilarité > X percentile
                - "standard_deviation" : coupe si > X écarts-types
                - "interquartile" : coupe sur base interquartile
            breakpoint_threshold_amount: seuil de détection des ruptures
                - Pour "percentile" : valeur entre 0 et 100
                  Plus élevé = moins de coupures = chunks plus grands
                  Plus bas = plus de coupures = chunks plus petits
        """
        self.child_size = child_size
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount

        # Étape 1 — Splitter sémantique
        self.semantic_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount
        )

        # Étape 2 — Splitter enfants
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_size // 5,
            separators=["\n\n", "\n", " ", ""]
        )

        print(
            f"✅ SemanticParentChildSplitter initialisé\n"
            f"   → Méthode sémantique : {breakpoint_threshold_type} "
            f"({breakpoint_threshold_amount})\n"
            f"   → Taille enfants : {child_size} caractères"
        )

    def split(
        self,
        documents: list
    ) -> tuple:
        """
        Découpe les documents en deux étapes :
        1. Sémantique → Parents
        2. Parents → Enfants

        Args:
            documents: liste de documents LangChain

        Returns:
            Tuple (liste_parents, liste_enfants)
        """
        if not documents:
            raise ValueError("La liste de documents est vide.")

        all_parents = []
        all_children = []

        for doc in documents:
            # ── Étape 1 : Découpage sémantique ──
            semantic_chunks = self.semantic_splitter.split_documents([doc])

            print(
                f"   → {doc.metadata.get('source', 'inconnu')} : "
                f"{len(semantic_chunks)} chunk(s) sémantique(s)"
            )

            # ── Étape 2 : Chaque chunk sémantique = un Parent ──
            for semantic_chunk in semantic_chunks:

                # Génère un ID unique pour ce parent
                parent_id = str(uuid.uuid4())

                # Enrichit les métadonnées du parent
                semantic_chunk.metadata["parent_id"] = parent_id
                semantic_chunk.metadata["chunk_type"] = "parent"
                all_parents.append(semantic_chunk)

                # ── Étape 3 : Découpe le Parent en Enfants ──
                children = self.child_splitter.split_documents(
                    [semantic_chunk]
                )

                for child in children:
                    child.metadata["parent_id"] = parent_id
                    child.metadata["chunk_type"] = "child"
                    all_children.append(child)

        print(
            f"\n✅ Découpage terminé :\n"
            f"   → {len(all_parents)} parent(s) sémantique(s)\n"
            f"   → {len(all_children)} enfant(s) indexables"
        )

        return all_parents, all_children

    def update_child_size(self, new_child_size: int):
        """
        Met à jour la taille des chunks enfants.
        Réinstancie le child_splitter proprement.

        Args:
            new_child_size: nouvelle taille en caractères
        """
        self.child_size = new_child_size
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=new_child_size,
            chunk_overlap=new_child_size // 5,
            separators=["\n\n", "\n", " ", ""]
        )
        print(f"✅ Taille enfants mise à jour : {new_child_size}")

    def update_threshold(
        self,
        threshold_type: str = None,
        threshold_amount: float = None
    ):
        """
        Met à jour les paramètres de détection sémantique.
        Réinstancie le semantic_splitter proprement.

        Args:
            threshold_type: nouvelle méthode de détection
            threshold_amount: nouveau seuil
        """
        if threshold_type:
            self.breakpoint_threshold_type = threshold_type
        if threshold_amount is not None:
            self.breakpoint_threshold_amount = threshold_amount

        self.semantic_splitter = SemanticChunker(
            embeddings=self.semantic_splitter.embeddings,
            breakpoint_threshold_type=self.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.breakpoint_threshold_amount
        )
        print(
            f"✅ Seuil sémantique mis à jour : "
            f"{self.breakpoint_threshold_type} "
            f"({self.breakpoint_threshold_amount})"
        )