import os 
import fitz 
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams
from qdrant_client.models import PointStruct, SparseVectorParams, Prefetch, FusionQuery, Fusion, SparseVector
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from fastembed import TextEmbedding, SparseTextEmbedding

load_dotenv() 

def extract_text_from_pdf(file_bytes: bytes) -> list:
    """Uses PyMuPDF (fitz) to read raw byte streams in-memory and extract text."""
    chunks=[]
    document = fitz.open(stream=file_bytes, filetype="pdf")  
    for page_num, page in enumerate(document) : 
        chunks.append({
            "metadata" : {
                    "page_num" : page_num,
                }, 
            "page_content" : page.get_text().strip()
        })
    return chunks

def chunk_text(chunks : list, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Segments raw text strings into overlapping dictionary blocks containing metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=overlap
    )
    texts = [chunk["page_content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]  

    documents = splitter.create_documents(texts=texts, metadatas=metadatas)
    return documents 

def generate_embeddings(chunks: list) -> list:
    """Uses SentenceTransformers to turn text strings into 1024-dimension float vectors."""
    # print(TextEmbedding.list_supported_models()) 
    sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    text_embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    texts = [chunk.page_content for chunk in chunks]
    batch=64
    sparse_vector = list(sparse_embedding_model.embed(texts, batch_size=batch)) 
    dense_vector = list(text_embedding_model.embed(texts, batch_size=batch))

    Points=[]
    for index, chunk in enumerate(chunks) : 
        point_id = str(uuid.uuid4())

        payload = {
            "page_content" : chunk.page_content,
            "metadata" : chunk.metadata 
        }

        point = PointStruct(
            id=point_id, 
            vector={
                "sparse": {
                    "indices": sparse_vector[index].indices,
                    "values": sparse_vector[index].values
                },
                "dense" : dense_vector[index].tolist() 
            },
            payload = payload
        )
        Points.append(point)
    return Points 

def get_qdrant_client() : 
    client = QdrantClient(
        url= os.getenv("QDRANT_CLOUD_URL"), 
        api_key= os.getenv("QDRANT_API_KEY"),
        timeout=60,
        check_compatibility=False
    )
    return client 

def initialize_qdrant_collection(collection_name: str):
    """Connects via API to provision a brand-new private collection table on Qdrant Cloud."""
    client = get_qdrant_client() 
    
    try:
        client.create_collection(
            collection_name=collection_name, 
            vectors_config={
                "dense" : VectorParams(distance=Distance.COSINE,size=384),
            },
            sparse_vectors_config ={
                "sparse" : SparseVectorParams()
            }   
        )
        print(f"Created a new remote collection: {collection_name}")
    except Exception as e:
        if "already exists" in str(e).lower() or "conflict" in str(e).lower():
            print(f"Collection '{collection_name}' already active in cloud vault.")
            # return client.get_collection(collection_name)
        else:
            print(f"Notice during collection setup: {e}.")

def delete_qdrant_collection(collection_name : str) : 
    client = get_qdrant_client() 

    try : 
        client.delete_collection(collection_name)   
        print(f"Successfully destroyed collection: {collection_name}")
        return True
    except Exception as e : 
        print(f"Error during collection purge: {e}")
        return False 

def upsert_vectors(collection_name: str, points: list) -> bool :
    """Pushes the generated vector blocks and text metadata into the isolated cloud space."""
    client = get_qdrant_client() 

    initialize_qdrant_collection(collection_name) 

    try : 
        client.upsert(
            collection_name=collection_name, 
            points=points
        )
        print("Successful points uploaded to Qdrant Cloud.")
        return True 
    except Exception: 
        print("Failed to upload points to Qdrant Cloud.")
        return False

def query_vector_db(collection_name : str, query : str, top_match : int ) -> str : 

    sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
    dense_embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    client = get_qdrant_client() 

    sparse_vector = list(sparse_embedder.embed([query]))[0] 
    dense_vector = list(dense_embedder.embed([query]))[0].tolist() 

    points = client.query_points(
        collection_name=collection_name, 
        prefetch=[
            Prefetch(query=SparseVector(indices=sparse_vector.indices, values=sparse_vector.values), using="sparse", limit=top_match), 
            Prefetch(query=dense_vector, using="dense", limit=top_match), 
        ],
        limit=top_match,
        query=FusionQuery(fusion=Fusion.RRF)
    )
    return [point.payload["page_content"] for point in points.points]
