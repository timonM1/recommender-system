from fastapi import FastAPI, HTTPException
from api.recommender import recommend_user_user, recommend_item_item
from api.utils import load_user_data, load_chroma_collection, load_embedding_model

app = FastAPI()

# Carga los datos de user-user
df_ratings, similarity_df = load_user_data()

# Carga los datos de item-item
try:
    collection_synopsis = load_chroma_collection("synopsis")
    print(f"Colección 'synopsis' cargada correctamente con {collection_synopsis.count()} elementos")
except Exception as e:
    print(f"Error al cargar la colección 'synopsis': {str(e)}")
    print("La API se iniciará sin la funcionalidad de recomendación item-item")
    collection_synopsis = None

model = load_embedding_model()

@app.get("/user/{user_id}/user_recommendations/")
def get_user_recommendations(user_id: int, top_n: int = 5):
    user_id = f"user_{user_id}"
    recs = recommend_user_user(user_id, df_ratings, similarity_df, top_n)
    if not recs["neighbors"] and not recs["recommendations"]:
        raise HTTPException(status_code=404, detail="User not found or no data")
    return recs

@app.get("/user/{user_id}/item_recommendations/")
def get_item_item_recommendations(user_id: int, top_n: int = 5):
    user_id = f"user_{user_id}"
    if key not in df_ratings.index:
        raise HTTPException(status_code=404, detail="User not found")
    recs = recommend_item_item(user_id, df_ratings, collection_synopsis, model, top_n)
    return recs
