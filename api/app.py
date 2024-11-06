from fastapi import FastAPI, Form, Response, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain.prompts import PromptTemplate 
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_llm import LlamaCpp  
from preprocess import preprocess_query
import json

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Initialize model and database connection
local_llm_path = "ggml-model-Q4_K_M.gguf"
llm = LlamaCpp(model_path=local_llm_path, temperature=0.3, max_tokens=2048, top_p=1)
embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")  # Adjusted import
client = QdrantClient(url="http://localhost:6333", prefer_grpc=False)
db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")


few_shot_examples = [
    {
        "context": "The process of photosynthesis allows plants to convert light energy into chemical energy.",
        "question": "What is photosynthesis?",
        "answer": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll."
    },
    {
        "context": "Insulin is a hormone that regulates blood sugar levels.",
        "question": "What does insulin do?",
        "answer": "Insulin helps cells absorb glucose, reducing blood sugar levels, and is crucial for maintaining energy balance."
    }
]

# Format for the prompt
formatted_examples = "\n".join([f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}" 
    for ex in few_shot_examples
])

# Prompt template setup
prompt_template = """You are an expert in biomedical sciences. Use the following examples to answer the user's question.
{examples}

Context: {context}
Question: {question}

Helpful answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['examples', 'context', 'question'])

# Setup the retriever
retriever = db.as_retriever(search_kwargs={"k": 1})

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_response")
async def get_response(query: str = Form(...)):
    # Preprocess the query
    corrected_query, query_type = preprocess_query(query)
    
    # Use the  query in the QA chain
    context = ""
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "examples": formatted_examples}
    )
    
    # Get response using  query
    response = qa(corrected_query)
    answer = response['result']
    source_document = response['source_documents'][0].page_content
    doc = response['source_documents'][0].metadata['source']
    
    # Format and send JSON response
    response_data = json.dumps({"answer": answer, "source_document": source_document, "doc": doc})
    return Response(response_data, media_type="application/json")
