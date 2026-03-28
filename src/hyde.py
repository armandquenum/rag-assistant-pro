"""
HyDE — Hypothetical Document Embedding

Principe :
Au lieu de vectoriser la question directement,
on demande au LLM de générer une réponse hypothétique,
puis on vectorise cette réponse.

Avantage :
La réponse hypothétique est sémantiquement plus proche
des chunks du vector store que la question elle-même.
"""

from langchain_core.prompts import ChatPromptTemplate


class HyDE:
    """
    Classe implémentant le Hypothetical Document Embedding.

    Flux :
    Question → LLM → Réponse hypothétique → Embedding → FAISS
    """

    HYDE_PROMPT = """You are a generator of hypothetical answers.
    Your role is to generate a FICTITIOUS, PLAUSIBLE and DIRECT answer to the question.

    STRICT RULES:
    - You MUST always generate an answer, even if you do not know the correct answer
    - The answer may be made up but must sound natural and plausible
    - 1 to 2 sentences maximum
    - No phrases such as ‘I don’t know’ or ‘I don’t have any information’ or ‘I’ll try to answer’
    - Answer in the same language as the question
    - If the question is vague, invent a plausible context

    Examples:
    Question: ‘What does Lincoln eat?’
    Answer: ‘Lincoln mainly ate burgers.’

    Question: ‘When did he drop John from his name?’
    Answer: ‘He dropped John from his name in 1842 when he began his political career.’

    Question: {question}
    Hypothetical answer:"""

    def __init__(self, llm):
        """
        Args:
            llm: instance du LLM LangChain (ChatGroq)
        """
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            self.HYDE_PROMPT
        )
        print("✅ HyDE initialisé")

    def generate_hypothetical_document(
    self,
    question: str
) -> str:
        """
        Génère une réponse hypothétique pour une question.
        """
        if not question or not question.strip():
            raise ValueError("La question ne peut pas être vide.")

        # Filtre les questions parasites avant d'appeler le LLM
        phrases_parasites = [
            "voici",
            "reformulation",
            "je suis prêt",
            "pouvez-vous me poser",
            "here are",
            "following"
        ]
        question_lower = question.lower()
        if any(p in question_lower for p in phrases_parasites):
            print(f"   ⚠️ HyDE — Question parasite ignorée : '{question[:50]}'")
            return question  # Retourne la question telle quelle

        prompt_value = self.prompt.invoke({
            "question": question
        })

        response = self.llm.invoke(prompt_value)
        hypothetical_doc = response.content.strip()

        # Filtre les réponses absurdes du LLM
        reponses_absurdes = [
            "je suis prêt",
            "pouvez-vous",
            "je ne dispose pas",
            "je ne sais pas",
            "insufficient information",
            "je n'ai pas",
            "can you",
            "please provide",
            "i don't have",
            "i do not have",
            "i cannot",
            "je ne peux pas",
            "aucune information"
        ]
        hyp_lower = hypothetical_doc.lower()
        if any(r in hyp_lower for r in reponses_absurdes):
            print(
                f"   ⚠️ HyDE — LLM a refusé de répondre\n"
                f"   → Utilise la question originale comme fallback"
            )
            return question

        print(f"\n💭 HyDE — Réponse hypothétique :")
        print(f"   '{hypothetical_doc[:100]}...'")

        return hypothetical_doc

    def generate_batch(self, questions: list) -> list:
        """
        Génère des réponses hypothétiques pour plusieurs questions.

        Args:
            questions: liste de questions

        Returns:
            Liste de réponses hypothétiques
        """
        hypothetical_docs = []

        for question in questions:
            try:
                hyp_doc = self.generate_hypothetical_document(
                    question
                )
                hypothetical_docs.append(hyp_doc)
            except Exception as e:
                print(f"⚠️ HyDE erreur sur '{question}' : {e}")
                # En cas d'erreur, utilise la question originale
                hypothetical_docs.append(question)

        return hypothetical_docs