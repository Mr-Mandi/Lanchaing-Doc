import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

from utils import (
    convert_excel_to_csv,
    read_csv_lancing,
    convert_document_to_embeddings,
    connect_chromadb,
    question_to_db
)

load_dotenv()  # take environment variables from .env.


if __name__ == '__main__':
    os.getenv("OPENAI_API_KEY")
    df = convert_excel_to_csv()
    data = read_csv_lancing("Menu.csv")
    database = convert_document_to_embeddings(documents=data)
    text: str = ""
"""    llm = ChatOpenAI(model_name='gpt-3.5-turbo')
    chain = load_qa_chain(llm, chain_type="stuff")
    while text != "exit":
        text = input("Question: ")
        question_to_db(
            question=text,
            database=database,
            chain=chain
    )"""
