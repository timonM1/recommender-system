import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
import os

CHROMA_DB_PATH = os.path.join(os.getcwd(), "chroma_db")

def load_user_data():
    """
    Carga los datos de calificaciones de los usuarios desde un archivo CSV y 
    calcula la matriz de similitudes entre usuarios usando el coseno de similitud.

    Retorna:
        df_ratings (DataFrame): DataFrame con las calificaciones de los usuarios.
        similarity_df (DataFrame): DataFrame con la similitud de los usuarios.
    """
    try:
        df_ratings = pd.read_csv("data/user_ratings.csv", index_col=0)
        ratings_filled = df_ratings.fillna(0) # Rellenar las calificaciones faltantes con 0
        similarity = cosine_similarity(ratings_filled)
        similarity_df = pd.DataFrame(similarity, index=df_ratings.index, columns=df_ratings.index)
        return df_ratings, similarity_df
    except Exception as e:
        print(f"Error al cargar los datos de usuarios: {str(e)}")
        raise

def load_chroma_collection(name: str):
    """
    Carga una colección de ChromaDB desde un directorio persistente.

    Argumentos:
        name (str): Nombre de la colección de ChromaDB a cargar.

    Retorna:
        collection: La colección cargada desde ChromaDB.

    Lanza:
        Exception: Si ocurre un error al cargar la colección.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    try:
        collection = client.get_collection(name)
        return collection
    except Exception as e:
        print(f"Error al cargar la colección '{name}': {str(e)}")
        raise

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Carga y devuelve el modelo de SentenceTransformer para generar embeddings.

    Argumentos:
        model_name (str): Nombre del modelo de SentenceTransformer (por defecto 'all-MiniLM-L6-v2').

    Retorna:
        model: El modelo de SentenceTransformer cargado.
    """
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        print(f"Error al cargar el modelo '{model_name}': {str(e)}")
        raise
