# syntax=docker/dockerfile:1

FROM python:3.11

ENV BEARER_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6InBpbmVjb25lLWNoYXRncHQtaW50ZWdyYXRpb24iLCJpYXQiOjE1MTYyMzkwMjJ9.F71u4eSiCfZoHehkG9J5g3bQ4aTixkjFN7oalw3eM8Q

ENV GENAI_DATA_ASK_API_ENDPOINT=https://gen-ai-data-ask-api-app.jollydune-3cd7c339.eastus.azurecontainerapps.io/

WORKDIR /code

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8000


ENTRYPOINT ["gunicorn", "app:app",  "--bind", "0.0.0.0:8000", "--worker-class", "uvicorn.workers.UvicornWorker"]
