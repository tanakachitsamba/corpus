import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from dotenv import load_dotenv
import os

# Load environmental variables from .env file
load_dotenv()
# Now you can access the environmental variables as regular Python variables
api_key = os.getenv("OPENAI_KEY")


persist_directory = "db"

client = chromadb.Client(
    Settings(
        persist_directory=persist_directory,
        chroma_db_impl="duckdb+parquet",
    )
)

#Persisting DB to disk, putting it in the save folder db
client.persist()

# Start from scratch
##client.reset()

# Create a new chroma collection
collection_name = "peristed_collection"


# Using OpenAI Embeddings. This assumes you have the openai package installed
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key, # Replace with your own OpenAI API key
)

collection = client.create_collection(name=collection_name, embedding_function=openai_ef)
#Running Chroma using direct local API.
#No existing DB found in db, skipping load
#No existing DB found in db, skipping load
#/Users/antontroynikov/miniforge3/envs/chroma/lib/python3.9/site-packages/tqdm/auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
#from .autonotebook import tqdm as notebook_tqdm
# Add some data to the collection
collection.add(
    metadatas=[
        {"uri": "img1.png", "style": "style1"},
        {"uri": "img2.png", "style": "style2"},
        {"uri": "img3.png", "style": "style1"},
        {"uri": "img4.png", "style": "style1"},
        {"uri": "img5.png", "style": "style1"},
        {"uri": "img6.png", "style": "style1"},
        {"uri": "img7.png", "style": "style1"},
        {"uri": "img8.png", "style": "style1"},
    ],
    documents=["doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8"],
    ids=["id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8"],
)

results = collection.query(
    query_embeddings=[[1.1, 2.3, 3.2]],
    n_results=1
)

print(results)
