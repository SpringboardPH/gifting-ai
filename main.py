import os
import json
from supabase import create_client, Client
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_SERVICE_ROLE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
PRODUCTS_JSON_FILE = "gemini_products_clean_updated.json"

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Initialize Google Generative AI Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

def ingest_product_data():
    with open(PRODUCTS_JSON_FILE, 'r', encoding='utf-8') as f:
        products = json.load(f)

    docs = []
    ids = []
    BATCH_SIZE = 50
    total = len(products)
    print(f"Starting ingestion of {total} products into Supabase via LangChain...")

    for i, product in enumerate(products):
        product_id = product.get('id', str(uuid4()))
        title = product.get('title', '')
        description = product.get('description', '')
        tags = product.get('tags', [])
        price = product.get('price')
        image_url = product.get('image', '')
        product_url = product.get('url', '')
        shopify_product_id = product.get('shopify_product_id')

        text_to_embed = f"Title: {title}. Description: {description}. Tags: {', '.join(tags)}."
        metadata = {
            "id": product_id,
            "shopify_product_id": shopify_product_id,
            "title": title,
            "description": description,
            "tags": tags,
            "price": price,
            "image_url": image_url,
            "product_url": product_url
        }
        docs.append(Document(page_content=text_to_embed, metadata=metadata))
        ids.append(product_id)

        # Batch insert
        if (i + 1) % BATCH_SIZE == 0 or (i + 1) == total:
            batch_docs = docs[-BATCH_SIZE:]
            batch_ids = ids[-BATCH_SIZE:]
            try:
                SupabaseVectorStore.from_documents(
                    batch_docs,
                    embeddings,
                    client=supabase,
                    table_name="product_vectors",  # Use dedicated vector table
                    query_name="match_product_vectors",  # Ensure this function exists in Supabase
                    chunk_size=BATCH_SIZE,
                    ids=batch_ids
                )
                print(f"  Successfully ingested {i+1} of {total} products.")
            except Exception as e:
                print(f"  Error ingesting batch ending at {i+1}: {e}")

    print("\nIngestion complete!")
    # Optionally, verify total count directly from Supabase
    try:
        count_response = supabase.from_('product_vectors').select('*', count='exact').execute()
        if hasattr(count_response, 'count') and count_response.count is not None:
            print(f"Total products now in Supabase vector table: {count_response.count}")
        else:
            print("Could not retrieve total count from Supabase vector table.")
    except Exception as e:
        print(f"Error retrieving count from Supabase: {e}")

if __name__ == "__main__":
    ingest_product_data()