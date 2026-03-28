"""
Multi-Query Retrieval

Principe :
Au lieu d'une seule recherche FAISS,
on génère plusieurs reformulations de la question
et on effectue une recherche pour chacune.

Avantage :
On capture plus de chunks pertinents
en explorant différents angles sémantiques.
"""

from langchain_core.prompts import ChatPromptTemplate


class MultiQueryGenerator:
    """
    Classe générant plusieurs reformulations d'une question.

    Flux :
    Question → LLM → [Q1, Q2, Q3] → 3 recherches FAISS → Fusion
    """

    MULTI_QUERY_PROMPT = """GGenerates exactly {n_queries} rephrasings of the following question.

    STRICT RULES:
    - Answer ONLY with the rephrased questions
    - One rephrased question per line
    - No introduction, no explanation, no numbering
    - No phrases such as “Here are the rephrased questions”
    - Each line must be a complete question
    - Vary the vocabulary and structure
    - VERY IMPORTANT: keep EXACTLY the same language as the original question
    If the question is in English → rephrase in English
    If the question is in French → rephrase in French
    NEVER translate into another language

    Question: {question}

    Rephrases (same language, one per line):"""

    def __init__(self, llm, n_queries: int = 3):
        """
        Args:
            llm: instance du LLM LangChain
            n_queries: nombre de reformulations à générer
        """
        self.llm = llm
        self.n_queries = n_queries
        self.prompt = ChatPromptTemplate.from_template(
            self.MULTI_QUERY_PROMPT
        )
        print(f"✅ MultiQueryGenerator initialisé ({n_queries} requêtes)")

    def generate_queries(self, question: str) -> list:
        """
        Génère plusieurs reformulations d'une question.

        Args:
            question: question originale

        Returns:
            Liste de reformulations incluant la question originale
        """
        if not question or not question.strip():
            raise ValueError("La question ne peut pas être vide.")

        prompt_value = self.prompt.invoke({
            "question": question,
            "n_queries": self.n_queries
        })

        response = self.llm.invoke(prompt_value)

        # Parse les reformulations
        raw_lines = response.content.strip().split("\n")

        queries = []
        for line in raw_lines:
            line = line.strip()

            # Ignore les lignes vides
            if not line:
                continue

            # Ignore les lignes trop courtes
            if len(line) < 10:
                continue

            # Ignore les phrases d'introduction parasites
            phrases_parasites = [
                "voici",
                "voilà",
                "reformulation",
                "différentes",
                "question",
                "je suis prêt",
                "pouvez-vous",
                "here are",
                "here is",
                "following",
            ]
            line_lower = line.lower()
            if any(p in line_lower for p in phrases_parasites):
                continue

            # Nettoie les numérotations : "1.", "1)", "-", "*"
            import re
            line = re.sub(r'^[\d]+[.)]\s*', '', line)
            line = re.sub(r'^[-*•]\s*', '', line)
            line = line.strip()

            # Vérifie que c'est bien une question ou phrase complète
            if len(line) > 10:
                queries.append(line)

        # Garde max n_queries reformulations
        queries = queries[:self.n_queries]

        # Ajoute la question originale en premier
        all_queries = [question] + queries

        # Déduplique en gardant l'ordre
        seen = set()
        unique_queries = []
        for q in all_queries:
            q_normalized = q.lower().strip()
            if q_normalized not in seen:
                seen.add(q_normalized)
                unique_queries.append(q)

        print(f"\n🔀 Multi-query — {len(unique_queries)} requêtes :")
        for i, q in enumerate(unique_queries, 1):
            label = "original" if i == 1 else f"reformulation {i-1}"
            print(f"   {i}. [{label}] {q}")

        return unique_queries

    def set_n_queries(self, n: int):
        """
        Met à jour le nombre de reformulations.

        Args:
            n: nouveau nombre de reformulations
        """
        if n < 1:
            raise ValueError("n_queries doit être supérieur à 0.")
        self.n_queries = n
        print(f"✅ n_queries mis à jour : {n}")