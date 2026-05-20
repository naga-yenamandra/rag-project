import os
from anthropic import Anthropic
from dotenv import load_dotenv
from retriever import retrieve

load_dotenv()
client = Anthropic()
MODEL = 'claude-sonnet-4-5'

PROMPT_TEMPLATE = """You are a helpful research assistant answering questions about machine learning papers.

Answer the question using ONLY the context provided below. If the context doesn't contain enough information to answer, say "I don't have enough information to answer that based on the provided context."

When you use information from the context, cite the source at the end of the sentence in square brackets like [Source: filename.txt].

Context:
{context}

Question: {question}

Answer:"""

def format_context(results: list[tuple[float,dict]]) -> str:
    parts = []
    for score, chunk in results:
        parts.append(f"Source: {chunk['source']}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def answer(question: str) -> dict:
    results = retrieve(question)
    context = format_context(results)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = client.messages.create(
        model = MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "question": question,
        "answer": response.content[0].text,
        "sources": list(set(chunk['source'] for _, chunk in results)),
        "num_chunks_retrieved": len(results)
    }

if __name__ == "__main__":
    test_questions = [
        "What is attention in transformers?",
        "How does retrieval-augmented generation work?",
        "What is the ReAct framework?",
        "How is GPT-4 evaluated?",
        "What is Self-RAG?"
    ]

    for questions in test_questions:
        print("=" * 80)
        result = answer(questions)
        print(f"Q: {result['question']}\n")
        print(f"A: {result['answer']}\n")
        print(f"Sources: {', '.join(result['sources'])}")
        print(f"Chunks used:{result["num_chunks_retrieved"]}\n")
