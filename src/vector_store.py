import os
import json
import hashlib
from langchain_community.vectorstores import FAISS
from src.embeddings import EmbeddingModel


class VectorStore:
    """
    Classe responsable du stockage et de la recherche
    des vecteurs avec FAISS.

    Architecture Parent-Child :
    - FAISS indexe les Enfants (petits chunks pour recherche précise)
    - Un dictionnaire JSON stocke les Parents (grands chunks pour contexte LLM)

    Sécurité :
    - Vérification d'intégrité par hash SHA256
    - Protection contre les fichiers pkl malveillants
    - Vérification du chemin de chargement
    """

    DEFAULT_SAVE_PATH = "vector_store"

    def __init__(self, embedding_model: EmbeddingModel):
        """
        Args:
            embedding_model: instance de EmbeddingModel
        """
        self.embedding_model = embedding_model
        self.embeddings = embedding_model.get_embeddings()
        self.vector_store = None

        # Dictionnaire des parents : {parent_id: Document}
        self.parents_store = {}

    # ─────────────────────────────────────────
    # Méthodes privées — sécurité
    # ─────────────────────────────────────────

    def _compute_hash(self, file_path: str) -> str:
        """
        Calcule le hash SHA256 d'un fichier.

        Args:
            file_path: chemin vers le fichier

        Returns:
            Hash SHA256 en hexadécimal
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def _save_checksum(self, save_path: str):
        """
        Calcule et sauvegarde le hash du fichier pkl.

        Args:
            save_path: dossier contenant index.pkl
        """
        pkl_path = os.path.join(save_path, "index.pkl")
        checksum_path = os.path.join(save_path, "checksum.txt")

        file_hash = self._compute_hash(pkl_path)

        with open(checksum_path, "w") as f:
            f.write(file_hash)

        print(f"🔐 Checksum sauvegardé : {file_hash[:20]}...")

    def _verify_checksum(self, load_path: str):
        """
        Vérifie l'intégrité du fichier pkl avant chargement.

        Args:
            load_path: dossier contenant index.pkl

        Raises:
            FileNotFoundError: si checksum.txt est absent
            ValueError: si le hash ne correspond pas
        """
        pkl_path = os.path.join(load_path, "index.pkl")
        checksum_path = os.path.join(load_path, "checksum.txt")

        if not os.path.exists(checksum_path):
            raise FileNotFoundError(
                "⚠️ Fichier checksum.txt introuvable. "
                "Le vector store a peut-être été créé sans sécurité."
            )

        with open(checksum_path, "r") as f:
            saved_hash = f.read().strip()

        current_hash = self._compute_hash(pkl_path)

        if current_hash != saved_hash:
            raise ValueError(
                "⚠️ ALERTE SÉCURITÉ : Le fichier index.pkl a été "
                "modifié depuis la dernière sauvegarde. "
                "Chargement annulé."
            )

        print(f"🔐 Intégrité vérifiée : hash OK")

    def _verify_path(self, path: str):
        """
        Vérifie que le chemin de chargement est autorisé.

        Args:
            path: chemin à vérifier

        Raises:
            ValueError: si le chemin est suspect
        """
        authorized_base = os.path.abspath(".")
        requested_path = os.path.abspath(path)

        if not requested_path.startswith(authorized_base):
            raise ValueError(
                f"⚠️ Chemin non autorisé : {path}."
            )

    # ─────────────────────────────────────────
    # Gestion des parents
    # ─────────────────────────────────────────

    def _save_parents(self, save_path: str):
        """
        Sauvegarde le dictionnaire des parents en JSON.

        Args:
            save_path: dossier de sauvegarde
        """
        parents_path = os.path.join(save_path, "parents.json")

        # Sérialise les parents en JSON
        serialized = {}
        for parent_id, doc in self.parents_store.items():
            serialized[parent_id] = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }

        with open(parents_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)

        print(
            f"💾 {len(self.parents_store)} parent(s) "
            f"sauvegardé(s) → parents.json"
        )

    def _load_parents(self, load_path: str):
        """
        Charge le dictionnaire des parents depuis JSON.

        Args:
            load_path: dossier de chargement
        """
        from langchain_core.documents import Document

        parents_path = os.path.join(load_path, "parents.json")

        if not os.path.exists(parents_path):
            raise FileNotFoundError(
                f"Fichier parents.json introuvable : {parents_path}"
            )

        with open(parents_path, "r", encoding="utf-8") as f:
            serialized = json.load(f)

        self.parents_store = {
            parent_id: Document(
                page_content=data["page_content"],
                metadata=data["metadata"]
            )
            for parent_id, data in serialized.items()
        }

        print(
            f"📂 {len(self.parents_store)} parent(s) "
            f"chargé(s) depuis parents.json"
        )

    # ─────────────────────────────────────────
    # Méthodes publiques
    # ─────────────────────────────────────────

    def create_from_chunks(self, parents: list, children: list):
        """
        Crée le vector store FAISS à partir des enfants
        et stocke les parents dans le dictionnaire.

        Args:
            parents: liste de Documents parents
            children: liste de Documents enfants

        Returns:
            self — pour chaîner les méthodes
        """
        if not children:
            raise ValueError("La liste d'enfants est vide.")

        if not parents:
            raise ValueError("La liste de parents est vide.")

        print(
            f"⏳ Création du vector store...\n"
            f"   → {len(parents)} parent(s)\n"
            f"   → {len(children)} enfant(s) à indexer"
        )

        # Indexe les enfants dans FAISS
        self.vector_store = FAISS.from_documents(
            documents=children,
            embedding=self.embeddings
        )

        # Stocke les parents dans le dictionnaire
        self.parents_store = {
            doc.metadata["parent_id"]: doc
            for doc in parents
        }

        print(f"✅ Vector store créé avec succès")
        return self

    def add_incremental(self, parents: list, children: list):
        """
        Ajoute de nouveaux parents et enfants
        à un vector store existant.

        Args:
            parents: nouveaux Documents parents
            children: nouveaux Documents enfants
        """
        if self.vector_store is None:
            raise ValueError(
                "Vector store non initialisé. "
                "Appelez d'abord create_from_chunks()."
            )

        # Ajoute les enfants à FAISS
        self.vector_store.add_documents(children)

        # Ajoute les parents au dictionnaire
        new_parents = {
            doc.metadata["parent_id"]: doc
            for doc in parents
        }
        self.parents_store.update(new_parents)

        print(
            f"✅ Ajout incrémental :\n"
            f"   → {len(parents)} parent(s) ajouté(s)\n"
            f"   → {len(children)} enfant(s) indexé(s)"
        )

    def save(self, path: str = None):
        """
        Sauvegarde le vector store FAISS et les parents
        avec vérification d'intégrité.

        Args:
            path: dossier de sauvegarde
        """
        if self.vector_store is None:
            raise ValueError("Aucun vector store à sauvegarder.")

        save_path = path or self.DEFAULT_SAVE_PATH
        os.makedirs(save_path, exist_ok=True)

        # Sauvegarde FAISS
        self.vector_store.save_local(save_path)
        print(f"✅ FAISS sauvegardé : {save_path}/")

        # Sauvegarde les parents en JSON
        self._save_parents(save_path)

        # Sauvegarde le hash pour sécurité
        self._save_checksum(save_path)

    def load(self, path: str = None):
        """
        Charge le vector store FAISS et les parents
        après vérification de sécurité.

        Args:
            path: dossier de chargement

        Returns:
            self — pour chaîner les méthodes
        """
        load_path = path or self.DEFAULT_SAVE_PATH

        # Vérification 1 — chemin autorisé
        self._verify_path(load_path)

        # Vérification 2 — dossier existant
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"Vector store introuvable : {load_path}"
            )

        # Vérification 3 — intégrité
        self._verify_checksum(load_path)

        # Charge FAISS
        self.vector_store = FAISS.load_local(
            load_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # Charge les parents
        self._load_parents(load_path)

        print(f"✅ Vector store chargé depuis : {load_path}/")
        return self

    def exists(self, path: str = None) -> bool:
        """
        Vérifie si un vector store sauvegardé existe.

        Args:
            path: dossier à vérifier

        Returns:
            True si le vector store existe
        """
        check_path = path or self.DEFAULT_SAVE_PATH
        return (
            os.path.exists(check_path) and
            os.path.exists(os.path.join(check_path, "index.faiss")) and
            os.path.exists(os.path.join(check_path, "parents.json"))
        )

    def get_store(self):
        """
        Retourne le vector store FAISS.

        Returns:
            Instance FAISS LangChain
        """
        if self.vector_store is None:
            raise ValueError(
                "Vector store non initialisé. "
                "Appelez create_from_chunks() ou load()."
            )
        return self.vector_store

    def get_parents_store(self) -> dict:
        """
        Retourne le dictionnaire des parents.

        Returns:
            Dict {parent_id: Document}
        """
        return self.parents_store
