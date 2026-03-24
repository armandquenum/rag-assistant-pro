from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
import os


class DocumentLoader:
    """
    Classe responsable du chargement des documents.
    Supporte les formats : PDF, TXT, DOCX
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader,
    }

    def __init__(self):
        self.loaded_documents = []

    def load_file(self, file_path: str) -> list:
        """
        Charge un seul fichier selon son extension.
        
        Args:
            file_path: chemin vers le fichier
            
        Returns:
            Liste de documents LangChain
            
        Raises:
            ValueError: si l'extension n'est pas supportée
            FileNotFoundError: si le fichier n'existe pas
        """
        # Vérifier que le fichier existe
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")

        # Récupérer l'extension
        extension = os.path.splitext(file_path)[1].lower()

        # Vérifier que l'extension est supportée
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Extension '{extension}' non supportée. "
                f"Extensions acceptées : {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )

        # Charger avec le bon loader
        loader_class = self.SUPPORTED_EXTENSIONS[extension]
        loader = loader_class(file_path)
        documents = loader.load()

        return documents

    def load_multiple_files(self, file_paths: list) -> list:
        """
        Charge plusieurs fichiers d'un coup.
        
        Args:
            file_paths: liste de chemins vers les fichiers
            
        Returns:
            Liste combinée de tous les documents
        """
        all_documents = []

        for file_path in file_paths:
            try:
                documents = self.load_file(file_path)
                all_documents.extend(documents)
                print(f"✅ Chargé : {os.path.basename(file_path)} "
                      f"({len(documents)} page(s))")
            except (FileNotFoundError, ValueError) as e:
                print(f"❌ Erreur sur {file_path} : {e}")

        self.loaded_documents = all_documents
        print(f"\n📚 Total : {len(all_documents)} page(s) chargée(s)")
        return all_documents

    def get_supported_extensions(self) -> list:
        """Retourne la liste des extensions supportées."""
        return list(self.SUPPORTED_EXTENSIONS.keys())