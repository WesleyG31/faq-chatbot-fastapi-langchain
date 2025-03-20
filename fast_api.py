# API dependencies
from fastapi import FastAPI
import chromadb
#from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

# Langchain dependencies / Models
import torch
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
#from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory

#uvicorn fast_api:app --reload


# 📌 Init FastApi
app = FastAPI()

# 📌 Embedding Model that we have created before
device = "cuda" if torch.cuda.is_available() else "cpu"
#embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={"device": "cuda"})
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en",model_kwargs={"device": device})
#embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-mistral-7b-instruct")

# LLM
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llama3.1"  # Modelo LLaMA 3.1

# 📌 Init ChromaDB / Read the vector_db
vectorstore=Chroma(persist_directory="chroma_db", embedding_function=embedding_model)
retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": 2})
#chroma_client = Chroma.PersistentClient(path="chroma_db")
#faq_collection = chroma_client.get_or_create_collection(name="faq_ecommerce")


# create the prompt template
prompt = """
You are an assistant for answering e-commerce FAQs. Use the retrieved context to answer user questions concisely and helpfully.
If you don't know the answer, say so.
Always answer in a professional tone.
Something the question wil be in spanish in that case give the answer in spanish.

Question: {question}
Context: {context}
Answer:
"""
prompt_template = ChatPromptTemplate.from_template(prompt)

#Set Chat with Ollama

llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_URL)


#Create the RAG

rag_chain=({"context": retriever| (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
            "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser())


### NO LLM ########

def search_faq(query, top_k=1):
    db_client = chromadb.PersistentClient(path="chroma_db")
    collection = db_client.get_or_create_collection(name="faq_ecommerce")
    #embedding_model_transformers = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    #query_embedding = embedding_model_transformers.encode(query, convert_to_numpy=True).tolist()

    query_embedding = embedding_model.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    if results["ids"]:
        best_match = results["metadatas"][0][0]  # Get the top result
        #print(results["metadatas"][0][0])
        return best_match["answer"]
    else:
        return "Sorry, I couldn't find an answer to your question."
    
### NO LLM ########



# 📌 API will get a JSON with "question"
class QueryRequest(BaseModel):
    question: str


# 📌 Endpoint 
@app.post("/ask")
async def ask_question(request: QueryRequest):

    
    response = rag_chain.invoke(request.question)
    #response = search_faq(request.question)
    clean_response = response.replace("\n", " ")
    return clean_response

# 📌 Endpoint de prueba
@app.get("/")
def home():
    return {"message": "E-commerce FAQ Chatbot API with LLaMA 3.1 is running!"}
