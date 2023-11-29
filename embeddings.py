import openai
import os

import pandas as pd
import ast
from scipy import spatial

openai.api_key = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_FILE = "base_embeddings.csv"
EMBEDDINGS_PROMT = """
You will be provided with some additional data about the conference. 
It contains information about conference, lectures and other related events. 
It structured as title and content, where ":" is a separator.
Content could be separated with ";"
"""

def generate_embedding(text):
    embedding = openai.Embedding.create(
        input=[text],
        model=EMBEDDING_MODEL
    )
    return embedding['data'][0]['embedding']

def load_embeddings():
    return pd.read_csv(EMBEDDING_FILE)

###################
## GENERATIION
###################

def inputs_from_file(file_path):
    return open(file_path, 'r')

def generate(inputs):
    input_vector_paris = []
    for input in inputs:
        embedding = generate_embedding(input)
        input_vector_paris.append([input, embedding])
    return input_vector_paris

def generate_to_file(inputs):
    input_vector_paris = generate(inputs)
    df = pd.DataFrame(input_vector_paris, columns=["text", "embedding"])
    df.to_csv(EMBEDDING_FILE, index=False)   

###################
## SEARCH
###################

def similarity_search(query: str, 
           source: pd.DataFrame=load_embeddings(), 
           relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
           top_n: int = 3
    ) -> tuple[list[str], list[float]]:
    query_embedding = generate_embedding(query)

    text_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, ast.literal_eval(row["embedding"])))
        for i, row in source.iterrows()
    ]
    text_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    text, relatednesses = zip(*text_and_relatednesses)
    return text[:top_n], relatednesses[:top_n]

def find_similar(input):
    return similarity_search(query=input, source=load_embeddings())

# print(get_embedding('Java'))
# generate_to_file(["Java", "C#", "Python"])
# print(find_similar("Island"))
# print(find_similar("Snake"))
# print(find_similar("WinForms"))
# print(find_similar("Programming Language"))

# inputs = inputs_from_file('tech_embeddings.csv')
# generate_to_file(inputs)

# print(similarity_search(query="OpenAI"))
