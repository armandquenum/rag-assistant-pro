from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:
    """
    Classe responsable de la génération des embeddings.
    
    Utilise sentence-transformers en local — pas besoin d'API key.
    Le modèle est téléchargé automatiquement au premier lancement.
    
    Modèle utilisé : all-MiniLM-L6-v2
    - Léger : 90 Mo
    - Rapide : idéal pour un projet perso
    - Performant : bon équilibre vitesse/qualité
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = None):
        """
        Args:
            model_name: nom du modèle HuggingFace à utiliser
                       Par défaut : all-MiniLM-L6-v2
        """
        self.model_name = model_name or self.DEFAULT_MODEL

        print(f"⏳ Chargement du modèle d'embedding : {self.model_name}")
        print(f"   (Premier lancement = téléchargement ~90Mo)")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        print(f"✅ Modèle d'embedding chargé avec succès")

    def get_embeddings(self):
        """
        Retourne l'objet embeddings LangChain.
        Utilisé par le VectorStore pour indexer les chunks.
        
        Returns:
            Objet HuggingFaceEmbeddings compatible LangChain
        """
        return self.embeddings

    def embed_query(self, query: str) -> list:
        """
        Transforme une question en vecteur.
        Utilisé par le Retriever pour chercher les chunks similaires.
        
        Args:
            query: question de l'utilisateur
            
        Returns:
            Vecteur numérique représentant la question
        """
        if not query or not query.strip():
            raise ValueError("La question ne peut pas être vide.")

        vector = self.embeddings.embed_query(query)
        print(f"✅ Question encodée — vecteur de dimension : {len(vector)}")
        return vector

    def get_model_name(self) -> str:
        """Retourne le nom du modèle utilisé."""
        return self.model_name