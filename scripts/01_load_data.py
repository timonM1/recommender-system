import pandas as pd

df = pd.read_csv('data/movies.tsv', sep='|', header=None, encoding='latin-1')

df.columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
              'unknown', 'action', 'adventure', 'animation', 'children', 'comedy',
              'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror',
              'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western']

df = df[['movie_id', 'title', 'action', 'adventure', 'animation', 'children', 'comedy',
         'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror',
         'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western']]

df = df.sample(n=200, random_state=42).reset_index(drop=True)


df.to_csv('data/df.csv', index=False)
print("Datos de películas cargados y 200 seleccionadas aleatoriamente. Guardado en data/df.csv")

