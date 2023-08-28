




# from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.llms import OpenAI
# from langchain.chains import VectorDBQA
# from langchain.document_loaders import TextLoader
# import os
# from chromadb.utils import embedding_functions
# import chromadb

# def get_database_dir():
#     current_script_directory = os.path.dirname(os.path.abspath(__file__))

#     projects_root_directory = os.path.dirname(current_script_directory)

#     database_directory = "Database"
#     database_path = os.path.join(projects_root_directory, database_directory)

#     print("Database path:", database_path)
#     return database_path

# def load_process_text_db(text_file):
#         # Load and process the text
#     loader = TextLoader(text_file)
#     documents = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     texts = text_splitter.split_documents(documents)
#     return texts

# def start_db():

#     client = chromadb.PersistentClient(path=get_database_dir())

#     client.heartbeat() # returns a nanosecond heartbeat. Useful for making sure the client remains connected.
#     return client

# def add_to_database():
#     # create simple ids
#     ids = [str(i) for i in range(1, len(docs) + 1)]

#     # add data
#     example_db = Chroma.from_documents(docs, embedding_function, ids=ids)
#     docs = example_db.similarity_search(query)
#     print(docs[0].metadata

# # persist_directory = get_database_dir()
# # texts = load_process_text_db()

# # embedding = embedding_functions.DefaultEmbeddingFunction()
# # # embedding = OpenAIEmbeddings()
# # vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)

# # vectordb.persist()
# # vectordb = None

# # #load the persisted database from disk
# # vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# # collection = client.create_collection(name="my_collection")
# # qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectordb)

# # query = "What did the president say about Ketanji Brown Jackson"
# # qa.run(query)

# # # To cleanup, you can delete the collection
# # vectordb.delete_collection()
# # vectordb.persist()

# # # Or just nuke the persist directory
# # !rm -rf db/
