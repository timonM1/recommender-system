# Sistema de Recomendación - FastAPI

## Descripción

Este proyecto es una API REST construida con FastAPI para realizar recomendaciones tanto **user-user** como **item-item**:

- **User-User**: Basado en la similitud del coseno entre usuarios, con condiciones mínimas de similitud y diferencia en las películas vistas.
- **Item-Item**: Utilizando _embeddings_ de sinopsis generados por OpenAI y almacenados en una base de datos vectorial con **ChromaDB**.

## Estructura del Proyecto

```plaintext
recommender-system/
├── api/               # Archivos principales de la API
│   ├── main.py
│   ├── recommender.py # Logica de negocio
│   ├── utils.py
├── chroma_db/         # Directorio donde se guarda la base de datos local de Chroma
├── data/              # Archivos fuente (CSV, TSV) utilizados por los scripts
├── scripts/           # Scripts para carga inicial y generación de datos
│   ├── 01_load_data.py
│   ├── 02_generate_synopsis.py
│   ├── 03_generate_embeddings.py
│   └── 04_generate_users.py
├── requirements.txt     # Lista de dependencias del proyecto
└── .gitignore          # Archivos y carpetas a ignorar por git
```

## Requisitos

- Python 3.10+
- FastAPI
- Uvicorn
- ChromaDB
- Numpy
- Pandas
- Scikit-learn
- python-dotenv

## Ejecución de la API

### Creación del entorno virtual

python -m venv venv

### Activar entorno virtual

Windows:

- .\venv\Scripts\activate

Mac/Linux:

- source .venv/bin/activate

### Instalación de dependencias

pip install -r requirements.txt

### Configuración de .env (solo si deseas generar nuevas sinopsis)

OPENAI_API_KEY=tu_clave_de_api

### Inicialización de Datos

Si deseas reiniciar la base de datos o empezar desde cero, ejecuta los siguientes scripts en este orden:

- python/scripts/01_load_data.py -> Carga y filtra 200 películas desde movies.tsv
- python/scripts/02_generate_synopsis.py -> Genera sinopsis usando la API de OpenAI
- python/scripts/03_generate_embeddings.py -> Crea embeddings, genera la tabla 'synopsis' e inyecta los datos en ChromaDB
- python/scripts/04_generate_users.py -> Crea 30 usuarios y les asigna de 20 a 50 películas con ratings aleatorios (1 a 5)

### Ejecutar el servidor

uvicorn api.main:app --reload

### Endpoints principales

Ingresar los siguientes endpoints en alguna plataforma / app, para las consulas como postaman o thurner, Las consultas son de tipo GET.

- **Recomendación User-User**: /user/{user_id}/user_recommendations/
- **Recomendación Item-Item:**: /user/{user_id}/item_recommendations/

### Respuesta
