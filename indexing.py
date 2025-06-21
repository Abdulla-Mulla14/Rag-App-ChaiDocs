from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

# Load and chunk web content
loader = WebBaseLoader([
            'https://chaidocs.vercel.app/youtube/chai-aur-html/introduction/',
            'https://chaidocs.vercel.app/youtube/chai-aur-html/emmit-crash-course/',
            'https://chaidocs.vercel.app/youtube/chai-aur-html/html-tags/',
            'https://chaidocs.vercel.app/youtube/chai-aur-git/welcome/',
            'https://chaidocs.vercel.app/youtube/chai-aur-git/introduction/',
            'https://chaidocs.vercel.app/youtube/chai-aur-git/terminology/',
            'https://chaidocs.vercel.app/youtube/chai-aur-git/behind-the-scenes/',
            'https://chaidocs.vercel.app/youtube/chai-aur-git/branches/',
            'https://chaidocs.vercel.app/youtube/chai-aur-git/diff-stash-tags/',
            'https://chaidocs.vercel.app/youtube/chai-aur-git/managing-history/',
            'https://chaidocs.vercel.app/youtube/chai-aur-git/github/',
            'https://chaidocs.vercel.app/youtube/chai-aur-c/welcome/',
            'https://chaidocs.vercel.app/youtube/chai-aur-c/introduction/',
            'https://chaidocs.vercel.app/youtube/chai-aur-c/hello-world/',
            'https://chaidocs.vercel.app/youtube/chai-aur-c/variables-and-constants/',
            'https://chaidocs.vercel.app/youtube/chai-aur-c/data-types/',
            'https://chaidocs.vercel.app/youtube/chai-aur-c/operators/',
            'https://chaidocs.vercel.app/youtube/chai-aur-c/control-flow/',
            'https://chaidocs.vercel.app/youtube/chai-aur-c/loops/',
            'https://chaidocs.vercel.app/youtube/chai-aur-c/functions/',
            'https://chaidocs.vercel.app/youtube/chai-aur-django/welcome/',
            'https://chaidocs.vercel.app/youtube/chai-aur-django/getting-started/',
            'https://chaidocs.vercel.app/youtube/chai-aur-django/jinja-templates/',
            'https://chaidocs.vercel.app/youtube/chai-aur-django/tailwind/',
            'https://chaidocs.vercel.app/youtube/chai-aur-django/models/',
            'https://chaidocs.vercel.app/youtube/chai-aur-django/relationships-and-forms/',
            'https://chaidocs.vercel.app/youtube/chai-aur-sql/welcome/',
            'https://chaidocs.vercel.app/youtube/chai-aur-sql/introduction/',
            'https://chaidocs.vercel.app/youtube/chai-aur-sql/postgres/',
            'https://chaidocs.vercel.app/youtube/chai-aur-sql/normalization/',
            'https://chaidocs.vercel.app/youtube/chai-aur-sql/database-design-exercise/',
            'https://chaidocs.vercel.app/youtube/chai-aur-sql/joins-and-keys/',
            'https://chaidocs.vercel.app/youtube/chai-aur-sql/joins-exercise/',
            'https://chaidocs.vercel.app/youtube/chai-aur-devops/welcome/',
            'https://chaidocs.vercel.app/youtube/chai-aur-devops/setup-vpc/',
            'https://chaidocs.vercel.app/youtube/chai-aur-devops/setup-nginx/',
            'https://chaidocs.vercel.app/youtube/chai-aur-devops/nginx-rate-limiting/',
            'https://chaidocs.vercel.app/youtube/chai-aur-devops/nginx-ssl-setup/',
            'https://chaidocs.vercel.app/youtube/chai-aur-devops/node-nginx-vps/',
            'https://chaidocs.vercel.app/youtube/chai-aur-devops/postgresql-docker/',
            'https://chaidocs.vercel.app/youtube/chai-aur-devops/postgresql-vps/',
            'https://chaidocs.vercel.app/youtube/chai-aur-devops/node-logger/'
])
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Embed and store into Qdrant
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

vector_db = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding,
    url="http://localhost:6333",  # This matches your docker-compose setup
    collection_name="chai-docs-index"
)

print("✅ Indexing complete. Collection created.")
