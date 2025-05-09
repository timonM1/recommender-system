import numpy as np
import pandas as pd

def recommend_user_user(usuario_id: str, df_ratings: pd.DataFrame, user_similarity_df: pd.DataFrame, top_n: int):
    """
    Recomienda películas a un usuario basado en el filtrado colaborativo user-user.
    Devuelve un diccionario con:
      - neighbors: lista de IDs de usuarios vecinos más similares
      - recommendations: lista de diccionarios con título, calificación predicha y justificación
    """
    # Si el usuario no existe en el DataFrame
    if usuario_id not in df_ratings.index:
        return {"neighbors": [], "recommendations": []}

    # Ratings del usuario objetivo
    user_ratings = df_ratings.loc[usuario_id]

    # Buscar vecinos similares con datos suficientes
    filtered_similar_users = []
    for other_user in user_similarity_df.columns:
        if other_user == usuario_id:
            continue
        common = df_ratings.loc[[usuario_id, other_user]].dropna(axis=1, how='any')
        common_movies = common.shape[1]
        sim = user_similarity_df.at[usuario_id, other_user]
        if common_movies >= 8 and sim > 0.3:
            filtered_similar_users.append((other_user, sim))

    # Ordenar y seleccionar mejores vecinos
    filtered_similar_users.sort(key=lambda x: x[1], reverse=True)
    top_neighbors = [user for user, _ in filtered_similar_users[:top_n]]

    # Predecir rating para películas no vistas
    unseen = user_ratings[user_ratings.isna()].index.tolist()
    predictions = {}
    explanations = {}
    for movie in unseen:
        rating_sum = 0.0
        sim_sum = 0.0
        best_neighbor = None
        best_contrib = 0.0
        for neighbor in top_neighbors:
            rating = df_ratings.at[neighbor, movie]
            if not np.isnan(rating):
                sim = user_similarity_df.at[usuario_id, neighbor]
                weighted = sim * rating
                rating_sum += weighted
                sim_sum += sim
                if weighted > best_contrib:
                    best_contrib = weighted
                    best_neighbor = neighbor
        if sim_sum > 0:
            predictions[movie] = round(rating_sum / sim_sum, 2)
            explanations[movie] = best_neighbor

    # Formatear recomendaciones con justificación
    sorted_recs = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
    recs_list = []
    for title, score in sorted_recs:
        neighbor = explanations.get(title)
        if neighbor:
            reason = f"Based on similar user {neighbor} who rated this movie highly."
        else:
            reason = "No strong neighbor contribution found."
        recs_list.append({
            "title": title,
            "predicted_rating": score,
            "reason": reason
        })

    return {"neighbors": top_neighbors, "recommendations": recs_list}

def recommend_item_item(usuario_id: str, df_ratings, collection, model, top_n: int):
    """
    Genera recomendaciones item-item basadas en ChromaDB
    
    Args:
        user_id: ID del usuario
        df_ratings: DataFrame con calificaciones de usuarios
        collection: Colección de ChromaDB con los embeddings
        model: Modelo de embeddings
        top_n: Número de recomendaciones a generar
        
    Returns:
        Lista de tuplas (título, (score, origen))
    """
    
    if collection is None:
        print("Error: La colección ChromaDB no existe")
        return []
    
    count = collection.count()
    print(f"La colección contiene {count} elementos")
    
    if count == 0:
        print("Error: La colección ChromaDB está vacía")
        return []
    
    
    df_movies = pd.read_csv("data/df_synopsis.csv")

    # Obtener las peliculas valoradas con 4 o mas (favoritas) y vistas
    user_ratings = df_ratings.loc[usuario_id]
    favorites = user_ratings[user_ratings >= 4].dropna().index.tolist()
    seen = user_ratings.dropna().index.tolist()
    
    # Para cada película favorita, buscar similares en ChromaDB
    recommendations = {}
    for fav in favorites:
        sinopsis = df_movies.loc[df_movies['title'] == fav, 'synopsis'].values[0]
        embedding = model.encode(sinopsis).tolist()

        # Consultar ChromaDB
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_n,
            include=['documents','distances']
        )
        
        docs = results['documents'][0]
        dists = results['distances'][0]
        
        # Acumular recomendaciones
        for doc, dist in zip(docs, dists):
            if doc not in seen and doc not in favorites:
                score = 1 - (dist / 2) 
                # Mantener la mejor procedencia
                if doc not in recommendations or score > recommendations[doc][0]:
                    recommendations[doc] = (score, fav)


    # Ordenar y formatear salida
    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1][0], reverse=True)[:top_n]
    recs = []
    for title, (score, origin) in sorted_recs:
        recs.append({
            'title': title,
            'score': round(score,3),
            'origin': origin
        })
    return recs