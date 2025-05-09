import os
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Cargar la API Key desde .env
load_dotenv(".env")
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Cargar el dataset de películas preprocesado
df_movies = pd.read_csv("data/df.csv")
df_movies["synopsis"] = ""

# Función para generar sinopsis usando OpenAI
def generate_synopsis(title):
    prompt = f"Generate a short synopsis (1 paragraph) for a movie called '{title}'."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# Generar sinopsis para cada película
for i, row in df_movies.iterrows():
    try:
        synopsis = generate_synopsis(row["title"])
        df_movies.at[i, "synopsis"] = synopsis
        print(f"Generated synopsis for: {row['title']}")
        time.sleep(0.1)  # evitar bloqueos por rate limit
    except Exception as e:
        print(f"Error generating synopsis for {row['title']}: {e}")

df_movies.to_csv("data/df_synopsis.csv", index=False)
print("Todas las sinopsis fueron generadas y guardadas en 'data/df_synopsis.csv'")
