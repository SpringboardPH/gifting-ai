import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from fastapi.middleware.cors import CORSMiddleware

# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_SERVICE_ROLE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# Initialize Supabase client and embeddings
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

# Initialize vector store
vectorstore = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="product_vectors",
    query_name="match_product_vectors"
)

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    google_api_key=GEMINI_API_KEY,
    temperature=0.2
)

def build_query(age, relationship, hobbies, budget, urgency, gift_type):
    return (
        f"Suggest a gift for a {age}-year-old. "
        f"Relationship: {relationship}. "
        f"Hobbies: {hobbies}. "
        f"Budget: {budget}. "
        f"Urgency: {urgency}. "
        f"Gift type: {gift_type}."
    )

def get_llm_refined_query(prompt):
    system_prompt = (
        "You are a helpful assistant for a gift recommendation system. "
        "Your first priority is to generate a search query that will retrieve products most closely related to the user's hobbies and interests. "
        "Given the following user profile and gift requirements, generate a concise, specific search query that would best match a product in a gift database. "
        "Otherwise, focus on keywords and phrases that would help retrieve the most relevant gift for the user's context. "
        "The search query should be suitable for semantic search in a product vector database. "
        "Be as specific as possible, and include relevant product types, recipient interests, and any constraints such as budget or urgency."
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)
    return response.content.strip() if hasattr(response, 'content') else prompt

def get_gift_suggestions(age, relationship, hobbies, budget, urgency, gift_type, k=5):
    user_prompt = build_query(age, relationship, hobbies, budget, urgency, gift_type)
    refined_query = get_llm_refined_query(user_prompt)
    results = vectorstore.similarity_search(refined_query, k=k)
    gift_data = []
    for doc in results:
        meta = doc.metadata
        gift_data.append({
            "title": meta.get("title", "Unknown"),
            "price": meta.get("price", "N/A"),
            "url": meta.get("product_url", "N/A"),
            "image_url": meta.get("image_url", "")
        })
    return {
        "original_prompt": user_prompt,
        "refined_query": refined_query,
        "suggestions": gift_data
    }


class GiftRequest(BaseModel):
    age: str
    relationship: str
    hobbies: str
    budget: str
    urgency: str
    gift_type: str
    k: int = 5

@app.post("/suggest-gift")
def suggest_gift(request: GiftRequest):
    try:
        result = get_gift_suggestions(
            request.age,
            request.relationship,
            request.hobbies,
            request.budget,
            request.urgency,
            request.gift_type,
            k=request.k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))