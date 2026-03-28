# 🤖 RAG Assistant Pro

Un système de questions-réponses intelligent basé sur vos documents,
combinant découpage sémantique, architecture Parent-Child, HyDE,
Multi-Query et reranking CrossEncoder.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-latest-red)
![FAISS](https://img.shields.io/badge/FAISS-latest-orange)
![RAGAS](https://img.shields.io/badge/RAGAS-0.78-brightgreen)

## 🚀 Démo en ligne

👉 **[Accéder à l'application](https://rag-assistant-pro-4stm8dfyuvrvyqzk8r4z3e.streamlit.app/)**

## 📋 Présentation

RAG Assistant Pro est une application de **Retrieval-Augmented Generation**
qui permet de poser des questions en langage naturel sur vos documents
PDF, TXT et DOCX — en français ou en anglais.

### Ce qui le rend unique

- **Découpage sémantique** — les documents sont découpés aux ruptures
  de sens, pas arbitrairement par nombre de caractères
- **Architecture Parent-Child** — recherche précise sur les petits chunks,
  contexte large fourni au LLM
- **HyDE** — génère une réponse hypothétique pour améliorer la recherche
- **Multi-Query** — génère plusieurs reformulations pour maximiser le recall
- **Reranking CrossEncoder** — reclassement sémantique profond des résultats
- **Support multilingue** — documents et questions en 50 langues
- **Sécurité SHA256** — vérification d'intégrité du vector store
- **Historique conversationnel** — le RAG se souvient du contexte

## 🏗️ Architecture
```
Documents (PDF / TXT / DOCX)
         ↓
SemanticChunker        ← détecte les ruptures de sens
         ↓
ParentChildSplitter    ← Parents (contexte) + Enfants (recherche)
         ↓
FAISS Vector Store     ← indexe les Enfants (similarité cosinus)
         ↓
Question utilisateur
         ↓
Multi-Query            ← génère N reformulations
         ↓
HyDE                   ← génère des réponses hypothétiques
         ↓
Retriever FAISS        ← N+1 recherches fusionnées
         ↓
Parent Mapping         ← remonte aux Parents contextuels
         ↓
CrossEncoder Reranker  ← reclasse par pertinence réelle
         ↓
LLM Groq (LLaMA 3)    ← génère la réponse
         ↓
Réponse + Sources citées
```

## 🛠️ Stack technique

| Composant | Technologie |
|---|---|
| Interface | Streamlit |
| Orchestration | LangChain |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 (50 langues) |
| Similarité | FAISS — distance cosinus |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq — LLaMA 3.1 8B Instant |
| Découpage sémantique | LangChain SemanticChunker |
| HyDE | Hypothetical Document Embedding |
| Multi-Query | Reformulation automatique des questions |

## 📊 Évaluation RAGAS

Évaluation réalisée sur le **Stanford Question-Answer Dataset (Kaggle)**
avec **20 questions** de difficulté variée (medium + hard).

| Métrique | Score | Interprétation |
|---|---|---|
| Faithfulness | 0.61 ⚠️ | Hallucinations sur questions complexes |
| Answer Relevancy | 0.79 ✅ | Réponses pertinentes |
| Context Recall | 0.90 ✅ | Bons chunks trouvés dans 90% des cas |
| Context Precision | 0.83 ✅ | Peu de bruit dans les chunks |
| **Score global** | **0.78** | **Bon niveau pour un RAG M1** |

### Progression après optimisations

| Métrique | V1 (base) | V2 (optimisé) | Évolution |
|---|---|---|---|
| Context Recall | 0.67 | 0.90 | +0.23 ✅ |
| Context Precision | 0.50 | 0.83 | +0.33 ✅ |
| Answer Relevancy | 0.76 | 0.79 | +0.03 ✅ |
| Score global | 0.68 | 0.78 | +0.10 ✅ |

### Stack d'évaluation
- Dataset : Stanford Q&A Dataset (Kaggle)
- LLM évaluateur : Mistral Small + Google Gemini
- Embeddings : paraphrase-multilingual-MiniLM-L12-v2

## ⚙️ Installation

### 1. Clone le projet
```bash
git clone https://github.com/armandquenum/rag-assistant-pro
cd rag-assistant-pro
```

### 2. Crée un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Installe les dépendances
```bash
pip install -r requirements.txt
```

### 4. Configure les clés API

Crée un fichier `.env` à la racine :
```
GROQ_API_KEY=ta_clé_groq_ici
```

Obtiens une clé gratuite sur [groq.com](https://groq.com)

### 5. Lance l'application
```bash
streamlit run app.py
```

## 📖 Utilisation

1. **Charge tes documents** — PDF, TXT ou DOCX dans le panneau gauche
2. **Clique sur "Nouvel index"** — le système indexe tes documents
3. **Pose tes questions** — en langage naturel, en français ou en anglais
4. **Consulte les sources** — chaque réponse cite ses sources avec score
5. **Ajuste les paramètres** — Top K, HyDE, Multi-Query dans la sidebar

## 🌍 Support multilingue

Le RAG utilise le modèle `paraphrase-multilingual-MiniLM-L12-v2`
supportant **50 langues**.

Tu peux poser tes questions en français même si tes documents
sont en anglais — et vice versa.

## 📁 Structure du projet
```
rag-project/
├── src/
│   ├── document_loader.py    # Chargement PDF/TXT/DOCX
│   ├── text_splitter.py      # Découpage sémantique + Parent-Child
│   ├── embeddings.py         # Modèle d'embedding multilingue
│   ├── vector_store.py       # FAISS cosinus + Parents JSON + SHA256
│   ├── retriever.py          # HyDE + Multi-Query + Parent-Child
│   ├── reranker.py           # CrossEncoder + normalisation sigmoid
│   ├── hyde.py               # Hypothetical Document Embedding
│   ├── multi_query.py        # Reformulation automatique
│   └── rag_pipeline.py       # Orchestrateur principal
├── evaluation/
│   ├── evaluate.py           # Évaluateur RAGAS
│   ├── test_dataset.py       # Chargement du dataset Kaggle
│   └── results/
│       └── ragas_*.json      # Scores RAGAS sauvegardés
├── app.py                    # Interface Streamlit
├── requirements.txt
├── .env.example
└── README.md
```

## 🔑 Variables d'environnement

| Variable | Description | Obligatoire |
|---|---|---|
| `GROQ_API_KEY` | Clé API Groq pour le LLM | ✅ Oui |
| `MISTRAL_API_KEY` | Clé API Mistral pour l'évaluation RAGAS | ❌ Optionnel |
| `GOOGLE_API_KEY` | Clé API Google pour l'évaluation RAGAS | ❌ Optionnel |

## 👤 Auteur

**Daniel Quenum**
Master Intelligence Artificielle — ISTIC Rennes

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Daniel_Quenum-blue)](https://www.linkedin.com/in/daniel-quenum/)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black)](https://github.com/armandquenum)