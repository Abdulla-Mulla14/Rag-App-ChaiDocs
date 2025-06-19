from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


# Vector Embeddings - (yeh isliye banaya gaya hai kyunke jab user ki jab query aayegi tab hum usko jo hai numbers mein convert karenge.)
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# (making connection with the database. because when query comes from the user so that it can search from the database and provide the relevant information)
vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    embedding=embedding_model,
    collection_name="chai-docs-index"
)


# Take user query
query = input("> ")


# Vector Simalirity Search (query) in DB
search_result = vector_db.similarity_search(
    query=query
)


# SYSTEM_PROMPT and giving context to SYSTEM_PROMPT
context_text = "\n".join(
    f"""
    === Document #{i+1} ===
    Title: {doc.metadata.get('title', 'N/A')}
    Category: {doc.metadata.get('category', 'N/A')}
    Topic: {doc.metadata.get('topic', 'N/A')}
    URL: {doc.metadata.get('source', 'N/A')}
    Content:
    {doc.page_content.strip()}
    --------------------------
    """ for i, doc in enumerate(search_result)
)


SYSTEM_PROMPT = f"""
    ROLE:
    You are an expert programming assistant with deep knowledge of the ChaiDocs documentation.

    You help users by understanding technical queries and generating structured, reliable answers using only the provided documentation as reference.

    TASK:
    Based on the CONTEXT DOCUMENTS and USER QUESTION below, generate a helpful response. Focus on clarity, depth, and practical value.

    CONTEXT DOCUMENTS:
    {context_text}

    USER QUESTION:
    {query}

    📦 OUTPUT FORMAT (in JSON):

    {{
        "summary": "A short 1–2 sentence summary of the answer",
        "detailed_answer": "A thorough but clear explanation, based entirely on the docs",
        "code_examples": [
            {{
                "language": "e.g. python, html, bash, etc.",
                "description": "Brief explanation of the code",
                "code": "Actual code here"
            }}
        ],
        "key_points": [
            "Core concept or insight 1",
            "Core concept or insight 2",
            "Core concept or insight 3"
        ],
        "related_links": [
            {{
                "title": "Title from document",
                "url": "Exact source URL from metadata",
                "description": "What this link covers"
            }}
        ],
        "categories": ["Relevant category1", "Relevant category2"],
        "difficulty_level": "beginner / intermediate / advanced",
        "additional_resources": [
            "Extra tip or resource",
            "Another suggestion if useful"
        ]
    }}

    RULES:
    - Stick to the documents. Do not invent information.
    - Include real URLs from metadata.
    - Make sure the JSON is valid (no trailing commas).
    - Keep tone helpful and confident — not overly casual.

    Respond ONLY with valid JSON, nothing else.
"""



# Calling an LLM (and you can use any LLM (Gemini, ChatGPT, etc...))
chat_completion = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]
)

print(f"🤖: {chat_completion.choices[0].message.content}")