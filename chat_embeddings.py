from chat_openai import streamed_answer
from embeddings import similarity_search, EMBEDDINGS_PROMT

def similarity_search_content(query):
    return similarity_search(query)[0]

def streamed_answer_with_embeddings(query):
    additional_data = similarity_search_content(query)
    query_and_embeddings = f"""
        Here's the additional data {additional_data} 
        and here's user input {query}
    """
    return streamed_answer(query_and_embeddings)




