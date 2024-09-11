from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

class Vector_database:
    def __init__(self, doccument = None, vector_database = FAISS, embedding = HuggingFaceEmbeddings()):
        self.vector_database = vector_database
        self.embedding = embedding
        self.doccument = doccument
    
    def build_db(self,doccument):
        db = self.vector_database.from_documents(doccument, self.embedding)
        return db
    
    def get_retriever(self, search_type : str = "similarity", search_kwargs : dict = {"k": 10}):
        return self.db.as_retriever(search_type, search_kwargs)
