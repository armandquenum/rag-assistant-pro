# 🤖 RAG Assistant Pro

Un système de questions-réponses intelligent basé sur vos documents,
combinant découpage sémantique, architecture Parent-Child et reranking.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-latest-red)
![FAISS](https://img.shields.io/badge/FAISS-latest-orange)

## 🚀 Démo en ligne

👉 **[Accéder à l'application](LIEN_STREAMLIT_ICI)**

## 📋 Présentation

RAG Assistant Pro est une application de **Retrieval-Augmented Generation**
qui permet de poser des questions en langage naturel sur vos documents
PDF, TXT et DOCX.

### Ce qui le rend unique

- **Découpage sémantique** — les documents sont découpés aux ruptures
  de sens, pas arbitrairement aux caractères
- **Architecture Parent-Child** — recherche précise sur les petits chunks,
  contexte large fourni au LLM
- **Reranking CrossEncoder** — reclassement sémantique profond des résultats
- **Sécurité SHA256** — vérification d'intégrité du vector store

## 🏗️ Architecture
```
Documents (PDF / TXT / DOCX)
         ↓
SemanticChunker        ← détecte les ruptures de sens
         ↓
ParentChildSplitter    ← Parents (contexte) + Enfants (recherche)
         ↓
FAISS Vector Store     ← indexe les Enfants
         ↓
Question utilisateur
         ↓
Retriever FAISS        ← cherche les Enfants similaires
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
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq — LLaMA 3.1 8B |
| Découpage sémantique | LangChain SemanticChunker |

## ⚙️ Installation

### 1. Clone le projet
```bash
git clone https://github.com/armandquenum/rag-assistant-pro
cd rag-project
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

### 4. Configure la clé API

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
3. **Pose tes questions** — en langage naturel dans le chat
4. **Consulte les sources** — chaque réponse cite ses sources

## 📁 Structure du projet
```
rag-project/
├── src/
│   ├── document_loader.py    # Chargement PDF/TXT/DOCX
│   ├── text_splitter.py      # Découpage sémantique + Parent-Child
│   ├── embeddings.py         # Modèle d'embedding local
│   ├── vector_store.py       # FAISS + stockage Parents JSON
│   ├── retriever.py          # Recherche Parent-Child
│   ├── reranker.py           # CrossEncoder reranking
│   └── rag_pipeline.py       # Orchestrateur principal
├── app.py                    # Interface Streamlit
├── requirements.txt
├── .env.example
└── README.md
```

## 🔑 Variables d'environnement

| Variable | Description | Obligatoire |
|---|---|---|
| `GROQ_API_KEY` | Clé API Groq pour le LLM | ✅ Oui |

## 👤 Auteur

**Armand Quenum**
Master Intelligence Artificielle — ISTIC Rennes

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Armand_Quenum-blue)](https://www.linkedin.com/in/armand-quenum/)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black)](https://github.com/armandquenum)
```
