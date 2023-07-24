#!/usr/bin/env python3
import typing
import logging
import openai
from chat_utils import ask
import os
import gradio as gr
import requests
from typing import Dict, Any, Optional, List
from enum import Enum
import json


BEARER_TOKEN: str = os.environ.get("BEARER_TOKEN")
GENAI_DATA_ASK_API_ENDPOINT: str = os.environ.get("GENAI_DATA_ASK_API_ENDPOINT")
assert BEARER_TOKEN != None
assert GENAI_DATA_ASK_API_ENDPOINT != None

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from a2wsgi import ASGIMiddleware

# from predict import predict
fast_app = FastAPI()

class Source(str, Enum):                                                   
    email = "email"                                                       
    file = "file"                                                         
    chat = "chat"                                                         
    sql = "sql"   

class Metadata(BaseModel):               
    source: Optional[Source] = None      
    source_id: Optional[str] = None     
    url: Optional[str] = None           
    created_at: Optional[str] = None     
    author: Optional[str] = None        
    database: Optional[str] = None      
    tables: Optional[str] = None         
    sql: Optional[str] = None           
       
class ChunkWithMetadata(BaseModel):                      
    text: str                                            
    metadata: Metadata                                   
                                                          
    def format_with_metadata(self) -> str:               
        if self.metadata.tables:                         
            return f"{self.metadata.tables}:{self.text}"  
        return f"{self.metadata.source_id}:{self.text}"    


class Answer(BaseModel):                                                                                   
    content: str
    metadata: List[ChunkWithMetadata]
    
def dispatch_payload(payload: Dict[str, Any]) -> Answer:
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {BEARER_TOKEN}"
    }
    #url: str = f"{GENAI_DATA_ASK_API_ENDPOINT}/ask"
    url: str = f"{GENAI_DATA_ASK_API_ENDPOINT}/ask?structured=true"
    response: requests.Response = requests.post(url=url, headers=headers, json=payload)
    status_code = response.status_code
    content = response.json()
    print(f"status_code: {status_code}")
    print(f"content:")
    print(json.dumps(content, indent=4))
    ask = content.get('answer')
    return Answer(**ask)

def create_payload(question: str) -> Dict[str, Any]:

    payload: Dict[str, Any] = {
        "ask": {
            "question": question
        }
    }
    return payload

def ask(question: str):
    payload = create_payload(question)
    response: Answer = dispatch_payload(payload)
    metadata: List[ChunkWithMetadata] = response.metadata
    # output = ''
    # for m in metadata:
    #     output = output + str(m) + "\n"
    #     #m.metadata.<author....>
    print(response.content)
    #print(output)
    return response.content

def split_response(input_text: str, history: List):
    history = history or []
    response = chatbot(input_text)
    # print(response)
    # if "(Sources:" in response:
    #     answer_response, sources = response.split("(Sources:", 1)
    #     print(answer_response)
    #     sources = "(Sources: " + sources
    #     print(sources)
    # elif "(source:" in response:
    #     answer_response, sources = response.split("(source:", 1)
    #     print(answer_response)
    #     sources = "(source: " + sources
    #     print(sources)
    # else: 
    #     answer_response, sources = response, ""
        
    # full_response = f"{input_text}\n\n{answer_response}\n\n"
    
    #history.append((input_text, answer_response))
    
    print(response)
    
    #return history, history
    #return answer_response
    return response

def chatbot(conversation):
    new_message = ask(conversation)
    #return "User: " + conversation + "\n\nSystem: " + new_message + "\n\n"
    return new_message

# --- fastapi /predict route ---


class Request(BaseModel):
    question: str


class Result(BaseModel):
    score: float
    title: str
    text: str


class Response(BaseModel):
    results: typing.List[Result]



class Source(str, Enum):
    email = "email"
    file = "file"
    chat = "chat"
    sql = "sql"

class Ask(BaseModel):
    question: str



@fast_app.post("/predict", response_model=Response)
async def predict_api(request: Request):
    results = ask(request.question)
    return Response(
        results=[
            Result(score=r["score"], title=r["title"], text=r["text"]) for r in results
        ]
    )


# --- gradio demo ---


def gradio_predict(question: str):
    results = ask(question)

    best_result = results[0]

    return f"{best_result['title']}\n\n{best_result['text']}", best_result["score"]


demo = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(
        label="Ask a question about the data", placeholder="What is BASEL III?"
    ),
    outputs=[gr.Textbox(label="Answer")], #, gr.Number(label="Score")],
    allow_flagging="never",
)


app = gr.mount_gradio_app(fast_app, demo, path="/")
#app = ASGIMiddleware(gr_app)



