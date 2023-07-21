from typing import Any, List, Dict, Tuple
import openai
import requests
import logging
import os
import json

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)
logger = logging.getLogger(__name__)

BEARER_TOKEN: str = os.environ.get("BEARER_TOKEN")
assert BEARER_TOKEN != None

GENAI_DATA_PLUGIN_API_ENDPOINT: str = os.environ.get("GENAI_DATA_PLUGIN_API_ENDPOINT")
assert GENAI_DATA_PLUGIN_API_ENDPOINT != None

GENAI_DATA_SQL_API_ENDPOINT: str = os.environ.get("GENAI_DATA_SQL_API_ENDPOINT")
assert GENAI_DATA_SQL_API_ENDPOINT != None

OPENAI_COMPLETIONMODEL_DEPLOYMENTID: str = os.environ.get("OPENAI_COMPLETIONMODEL_DEPLOYMENTID")


def query_sql_db(query_prompt: str) -> Dict[str, Any]:
    """
    Query vector database to retrieve chunk with user's input questions.
    """

    logger.info(f"query_sql_db: query_prompt={query_prompt}")

    url = f"{GENAI_DATA_SQL_API_ENDPOINT}/query"
    logger.info(f"query_database: url={url}")
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer {BEARER_TOKEN}",
    }
    data = {"query": query_prompt}

    logger.info(f"requesting answer...")
    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        result = response.json()
        # process the result
        return result
    else:
        raise ValueError(f"Error: {response.status_code} : {response.content}")

def query_vector_db(query_prompt: str) -> Dict[str, Any]:
    """
    Query vector database to retrieve chunk with user's input questions.
    """

    logger.info(f"query_vector_db: query_prompt={query_prompt}")

    url = f"{GENAI_DATA_PLUGIN_API_ENDPOINT}/query"
    logger.info(f"query_database: url={url}")
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer {BEARER_TOKEN}",
    }
    data = {"queries": [{"query": query_prompt, "top_k": 5}]}

    logger.info(f"requesting chunks...")
    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        result = response.json()
        # process the result
        return result
    else:
        raise ValueError(f"Error: {response.status_code} : {response.content}")


def apply_prompt_template(question: str) -> str:
    """
        A helper function that applies additional template on user's question.
        Prompt engineering could be done here to improve the result. Here I will just use a minimal example.
    """
    prompt = f"""
        By considering above input from me, answer the question: {question}
    """
    return prompt

def get_chat_completion(
    messages,
    model="gpt-3.5-turbo",  # use "gpt-4" for better results
    max_tokens=1024,
    temperature=0.7,
    deployment_id = None,
):
    """
    Generate a chat completion using OpenAI's chat completion API.

    Args:
        messages: The list of messages in the chat history.
        model: The name of the model to use for the completion. Default is gpt-3.5-turbo, which is a fast, cheap and versatile model. Use gpt-4 for higher quality but slower results.

    Returns:
        A string containing the chat completion.

    Raises:
        Exception: If the OpenAI API call fails.
    """
    # call the OpenAI chat completion API with the given messages
    # Note: Azure Open AI requires deployment id


    response = {}
    if deployment_id == None:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        response = openai.ChatCompletion.create(
            deployment_id = deployment_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
     )

    return response


def call_chatgpt_api(user_question: str, chunks: List[str]) -> Dict[str, Any]:
    """
    Call chatgpt api with user's question and retrieved chunks.
    """
    # Send a request to the GPT-3 API
    messages = list(
        map(lambda chunk: {
            "role": "user",
            "content": chunk
        }, chunks))
    question = apply_prompt_template(user_question)
    messages.append({"role": "user", "content": question})
    response = get_chat_completion(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
        deployment_id=OPENAI_COMPLETIONMODEL_DEPLOYMENTID)
    
    return response


Chunks = List[str]
Metadata = List[Dict[str, str]]

def parse_vector_results(chunks_response: Dict[str, Any]) -> Tuple[Chunks, Metadata]:
    chunks = []
    metadata = []
    for result in chunks_response["results"]:
        for inner_result in result["results"]:
            chunks.append(inner_result["text"])
            metadata.append(inner_result["metadata"])

    return chunks, metadata

def parse_sql_results(sql_response: Dict[str, Any]) -> Tuple[Chunks, Metadata]:
    chunks = []
    metadata = []

    result = sql_response["result"]
    answer = result.get("Answer", None)
    tables = result.get("Tables", None)
    columns = result.get("Columns", None)

    chunks = [answer]
    metadata = [
        {
            "tables": tables,
            "columns": columns
        }
    ]
    return chunks, metadata

def ask(user_question: str) -> Dict[str, Any]:
    """
    Handle user's questions.
    """

    # Get chunks from vector db.
    chunks_response = query_vector_db(user_question)
    vector_chunks, vector_metadata = parse_vector_results(chunks_response)
    print("vector database")
    print("chunks")
    print(json.dumps(vector_chunks, indent=4))
    print("metadata")
    print(json.dumps(vector_metadata, indent=4))

    # Get chunks from database.
    sql_response = query_sql_db(user_question)
    sql_chunks, sql_metadata = parse_sql_results(sql_response)
    print("sql database")
    print("chunks")
    print(json.dumps(sql_chunks, indent=4))
    print("metadata")
    print(json.dumps(sql_metadata, indent=4))

    chunks = vector_chunks + sql_chunks
    metadata = vector_metadata + sql_metadata

    completion_response = call_chatgpt_api(user_question, chunks)
    #logging.info("Response: %s", completion_response)
    
    content = completion_response["choices"][0]["message"]["content"]

    #return ""
    return content, metadata
