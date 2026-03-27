"""
Dataset de test basé sur le Kaggle Question-Answer Dataset.

Structure du dataset Kaggle :
- Fichiers Q/R : S08/S09/S10_question_answer_pairs.txt
- Articles     : text_data/SXX_setY_aZ.txt

Stratégie :
1. Charge les Q/R depuis les fichiers .txt
2. Mappe chaque question à son article source
3. Filtre sur les questions faciles avec réponses courtes
4. Retourne un dataset propre pour RAGAS
"""

import os
import csv
from collections import defaultdict
import pandas as pd


# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────

KAGGLE_DIR = "evaluation/kaggle_dataset"
TEXT_DATA_DIR = os.path.join(KAGGLE_DIR, "text_data")

QA_FILES = [
    os.path.join(KAGGLE_DIR, "S08_question_answer_pairs.txt"),
    os.path.join(KAGGLE_DIR, "S09_question_answer_pairs.txt"),
    os.path.join(KAGGLE_DIR, "S10_question_answer_pairs.txt"),
]

MAX_QUESTIONS_PER_ARTICLE = 2
MAX_TOTAL_QUESTIONS = 10


# ─────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────

def read_file_safely(file_path: str) -> pd.DataFrame:
    """
    Lit un fichier TSV avec gestion des encodages.
    Essaie utf-8-sig d'abord, puis latin-1.
    """
    try:
        return pd.read_csv(
            file_path,
            sep='\t',
            encoding='utf-8-sig',
            quoting=csv.QUOTE_MINIMAL
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            file_path,
            sep='\t',
            encoding='latin-1'
        )


# ─────────────────────────────────────────
# Chargement du dataset
# ─────────────────────────────────────────

def load_kaggle_dataset(
    max_questions: int = MAX_TOTAL_QUESTIONS,
    difficulty: str = "easy"
) -> list:
    """
    Charge et filtre les questions du dataset Kaggle.

    Args:
        max_questions: nombre max de questions à charger
        difficulty: niveau de difficulté (easy/medium/hard)

    Returns:
        Liste de dicts {question, ground_truth, article_file, document}
    """
    dataset = []
    articles_used = defaultdict(int)

    for qa_file in QA_FILES:
        if len(dataset) >= max_questions:
            break

        if not os.path.exists(qa_file):
            print(f"⚠️ Fichier manquant : {qa_file}")
            continue

        # Chargement
        df = read_file_safely(qa_file)

        # Nettoyage des colonnes
        df.columns = [col.strip() for col in df.columns]

        # Nettoyage des données
        df = df.drop_duplicates()
        df = df.dropna()
        df = df.map(str.strip)  # ← assignation corrigée

        # Filtre difficulté
        df["DifficultyFromQuestioner"] = (
            df["DifficultyFromQuestioner"].str.lower()
        )
        df = df[df["DifficultyFromQuestioner"] == difficulty]

        # Filtre réponses oui/non
        df = df[~df['Answer'].str.lower().isin(
            ["yes", "no", "yes.", "no."]
        )]

        # Filtre réponses trop longues — max 10 mots
        df = df[
            df['Answer'].apply(lambda x: len(x.split())) <= 10
        ]

        # Filtre réponses trop courtes — min 2 mots
        df = df[
            df['Answer'].apply(lambda x: len(x.split())) >= 2
        ]

        # Itération propre avec iterrows
        for _, row in df.iterrows():
            if len(dataset) >= max_questions:
                break

            question = str(row["Question"]).strip()
            answer = str(row["Answer"]).strip()
            article_file = str(row["ArticleFile"]).strip()

            # Limite par article
            if articles_used[article_file] >= MAX_QUESTIONS_PER_ARTICLE:
                continue

            # Vérifie que l'article existe
            article_path = os.path.join(
                TEXT_DATA_DIR,
                f"{article_file}.txt"
            )

            if not os.path.exists(article_path):
                continue

            dataset.append({
                "question": question,
                "ground_truth": answer,
                "article_file": article_file,
                "document": article_path
            })

            articles_used[article_file] += 1

    print(
        f"✅ Dataset chargé : {len(dataset)} questions\n"
        f"   → {len(articles_used)} article(s) différent(s)\n"
        f"   → Difficulté : {difficulty}"
    )

    return dataset


# ─────────────────────────────────────────
# Utilitaires dataset
# ─────────────────────────────────────────

def get_unique_documents(dataset: list) -> list:
    """Retourne la liste des documents uniques à indexer."""
    return list(set(item["document"] for item in dataset))


def get_questions(dataset: list) -> list:
    """Retourne la liste des questions."""
    return [item["question"] for item in dataset]


def get_ground_truths(dataset: list) -> list:
    """Retourne la liste des réponses de référence."""
    return [item["ground_truth"] for item in dataset]


def preview_dataset(dataset: list, n: int = 3):
    """Affiche un aperçu du dataset."""
    print(f"\n📋 Aperçu du dataset ({n} exemples) :")
    print("─" * 60)
    for i, item in enumerate(dataset[:n], 1):
        print(f"\n[{i}] Article  : {item['article_file']}")
        print(f"    Question : {item['question']}")
        print(f"    Réponse  : {item['ground_truth']}")
    print("─" * 60)


# ─────────────────────────────────────────
# Dataset par défaut
# ─────────────────────────────────────────

TEST_DATASET = load_kaggle_dataset(
    max_questions=MAX_TOTAL_QUESTIONS,
    difficulty="easy"
)