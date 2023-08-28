import chromadb
from chromadb.utils import embedding_functions

db_path = "PodcastBuddy/Database"
default_embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
# #uses base model and cpu
# # instruct_embeddings_gpu = embedding_functions.InstructorEmbeddingFunction(
# #     model_name="hkunlp/instructor-xl", device="cuda")
client = chromadb.PersistentClient(path=db_path)

def get_create_collection(collection_name):
    collection = client.get_or_create_collection(name=collection_name, embedding_function=default_embeddings)
    return collection

def add_to_collection(collection, text, metadata, ids):
    embeddings = default_embeddings(text)
    # print(embeddings)
    collection.add(
        documents=text,
        embeddings=embeddings,
        metadatas=metadata,
        ids=ids,
    )
    return collection
    

def similar_search_db(collection, query_embeddings, n_results, where):
    collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        where={"style": "style2"}
)