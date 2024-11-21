import os
import openai
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import AzureSearch
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_type = "azure"
openai.api_version = "2024-08-01-preview"

embeddings = OpenAIEmbeddings(deployment= "text-embedding-ada-002", chunk_size=1)

acs = AzureSearch(
    azure_search_endpoint= os.getenv('SEARCH_SERVICE_NAME'),
    azure_search_key= os.getenv('SEARCH_API_KEY'),
    index_name= os.getenv('SEARCH_INDEX_NAME'),
    embedding_function = embeddings.embed_query
)

def search(query) : 

    docs = acs.similarity_search_with_relevance_scores(
        query = query,
        k = 5 
    )
    
    result = docs[0][0].page_content

    return result

def assistant(query , context) :

    messages = [
        {"role" : "system" , "content" : "Assistant is a chatbot that helps you find the best wine for your taste."},
        {"role" : "user" , "content" : query},
        {"role" : "assistant" , "content" : context}
    ]

    response = openai.ChatCompletion.create(
        engine = 'gpt-4',
        messages = messages
    )

    result = response['choices'][0]['message']['content']

    return 'Hello' + result

class Body(BaseModel):
    query : str 

@app.post('/ask')
def ask(body : Body) :

    search_result = search(body.query)
    chat_bot_response = assistant(body.query , search_result)

    return {'response' : chat_bot_response} 

