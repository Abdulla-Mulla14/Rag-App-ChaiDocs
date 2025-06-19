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
context_text = ""
for doc in search_result:
    context_text += f"""
    Title: {doc.metadata.get('title', 'N/A')}
    Category: {doc.metadata.get('category', 'N/A')}
    Topic: {doc.metadata.get('topic', 'N/A')}
    URL: {doc.metadata.get('source', 'N/A')}
    Content: {doc.page_content}
    ---
    """

SYSTEM_PROMPT = f"""
    You are a helpful assistant that provides detailed answers about programming and development topics based on the Chai Docs documentation.

    Based on the following context documents, please answer the user's question and format your response as a JSON object with the following structure:

    {{
        "summary": "Brief overview of the answer in 1-2 sentences",
        "detailed_answer": "Comprehensive explanation of the topic",
        "code_examples": [
            {{
                "language": "html/python/javascript/sql/bash",
                "description": "What this code does",
                "code": "actual code here"
            }}
        ],
        "key_points": [
            "Important point 1",
            "Important point 2",
            "Important point 3"
        ],
        "related_links": [
            {{
                "title": "Link title from source",
                "url": "actual URL from metadata",
                "description": "Brief description of what this link covers"
            }}
        ],
        "categories": ["category1", "category2"],
        "difficulty_level": "beginner/intermediate/advanced",
        "additional_resources": [
            "Suggestion 1 for further learning",
            "Suggestion 2 for further learning"
        ]
    }}

    CONTEXT DOCUMENTS:
    {context_text}

    USER QUESTION: {query}

    Important guidelines:
    1. Extract actual URLs from the document metadata for the related_links section
    2. Include practical code examples when relevant
    3. Make the detailed_answer comprehensive but well-structured
    4. Ensure all JSON is properly formatted and valid
    5. Use the categories from the source documents
    6. Provide actionable key points

    Return ONLY the JSON response, no additional text before or after.
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