"""
Package evaluation : Pour évaluer RAG.
Regroupe les modules de chargement du dataset et d'évaluatio.
"""


from .test_dataset import TEST_DATASET, get_unique_documents, get_questions, get_ground_truths, preview_dataset
from .evaluate import run_evaluation



# Définit ce qui est exposé lors d'un "from src import *"
__all__ = [
    "TEST_DATASET",
    "get_unique_documents",
    "get_questions",
    "get_ground_truths",
    "preview_dataset",
    "run_evaluation"
]