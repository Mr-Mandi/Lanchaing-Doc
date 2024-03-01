import os
import uuid
import chromadb
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,

)
from langchain_community.document_loaders.csv_loader import CSVLoader

current_directory = os.getcwd()


def connect_chromadb():
    chroma_client = chromadb.HttpClient(
        host='localhost',
        port=8000
    )
    collection = chroma_client.get_collection(name="helpy")
    return chroma_client


def add_info_collection(documents: list, chroma_client: chromadb.HttpClient, collection):
    for doc in documents:
        collection.add(
            ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
        )


def convert_excel_to_csv() -> pd.DataFrame:
    df = pd.read_excel(f"{current_directory}/resources/Menu.xlsx")
    df.to_csv(
        f"{current_directory}/resources/Menu.csv",
        index=False
    )
    csv = pd.read_csv(f"{current_directory}/resources/Menu.csv")
    return csv


def read_csv_lancing(file_name: str) -> list:
    loader = CSVLoader(file_path=f"{current_directory}/resources/{file_name}")
    data = loader.load()
    return data


def convert_document_to_embeddings(documents: list):
    chroma_client = connect_chromadb()
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    collection = chroma_client.get_collection(name="helpy")
    if collection.count() == 0:
        add_info_collection(
            documents=documents,
            chroma_client=chroma_client,
            collection=collection
        )

    db = Chroma(
        client=chroma_client,
        collection_name="helpy",
        embedding_function=embedding_function,
    )
    return db


def question_to_db(question: str, database: Chroma, chain):
    responses = database.similarity_search(question, 3)
    response = chain.run(input_documents=responses, question=question)
    print(f"\nbot : {response}")
    """for response in responses:
        print(f"\n {response.page_content}")"""
