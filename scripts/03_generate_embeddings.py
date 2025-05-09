import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import shutil
import os


CHROMA_DB_PATH = os.path.join(os.getcwd(), "chroma_db")

if os.path.exists(CHROMA_DB_PATH):
    shutil.rmtree(CHROMA_DB_PATH)

df_movies = pd.read_csv("data/df_synopsis.csv")

model = SentenceTransformer('all-MiniLM-L6-v2')

# Generar los embeddings para las sinopsis de cada película
df_movies['embedding_synopsis'] = df_movies['synopsis'].apply(lambda x: model.encode(x).tolist())

# --- Inyectar embeddings en la base de datos ChromaDB ---
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Obtener la colección existente llamada 'synopsis' (o crearla si no existe)
collection = chroma_client.get_or_create_collection(
    name="synopsis",
    metadata={"hnsw:space": "cosine"}  
)

if collection is None:
    print("La colección no se ha creado correctamente.")
else:
    print("Colección 'synopsis' creada o cargada correctamente.")

# Insertar o actualizar los títulos, embeddings e IDs en la colección de ChromaDB
collection.add(
    documents=df_movies['title'].tolist(),
    embeddings=df_movies['embedding_synopsis'].tolist(),
    ids=df_movies['movie_id'].astype(str).tolist()
)
print("Embeddings inyectados en la colección 'synopsis' de ChromaDB.")

# DEBUG: comprueba que efectivamente se inyectaron vectores
print("Total de elementos en la colección 'synopsis' de ChromaDB:", collection.count())

