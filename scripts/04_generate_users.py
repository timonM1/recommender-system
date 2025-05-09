import pandas as pd
import numpy as np
import random

df_movies = pd.read_csv("data/df_synopsis.csv")

num_users = 30  
user_ids = [f'user_{i+1}' for i in range(num_users)] 
movie_titles = df_movies['title'].tolist() 

# Generar calificaciones aleatorias por usuario
ratings_data = {}  

for user in user_ids:
    seen_movies = random.sample(movie_titles, random.randint(20, 50))
    user_ratings = {movie: random.randint(1, 5) for movie in seen_movies}
    ratings_data[user] = user_ratings

# --- Convertir el diccionario de calificaciones en un DataFrame ---
# Filas: usuarios, Columnas: títulos de películas, Valores: calificaciones (NaN si no la vio)
df_ratings = pd.DataFrame.from_dict(ratings_data, orient='index')

# Ordenar las columnas (películas) alfabéticamente para mantener consistencia
df_ratings = df_ratings.sort_index(axis=1)

df_ratings.to_csv("data/user_ratings.csv")
print("Datos de usuarios y calificaciones generadas guardados en 'data/user_ratings.csv'.")
