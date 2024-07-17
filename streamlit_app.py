import os
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import bs4
__import__('pysqlite3')
from langchain.llms import OpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
import streamlit as st


st.title("✨🐆 Let's play with LangChain 🦜✨")

openai_api_key = os.environ.get("OPEN_AI_KEY")

# indexing: load data
# TODO: solve https issue
bs4_strainer = bs4.SoupStrainer(("h3", "a", "b", "i"))
loader = WebBaseLoader(
    web_paths=("http://shakespeare.mit.edu/midsummer/full.html",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

# indexing: split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# indexing: store
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))

# retrieval
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

retrieved_docs = retriever.invoke("Who is in love with Hermia?")

print(len(retrieved_docs))
print(retrieved_docs[0].page_content)

# chat interface

def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key, model="gpt-3.5-turbo-0125")
    st.info(llm(input_text))

with st.form('my-form'):
    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
    if submitted:
        generate_response(text)

# https://python.langchain.com/v0.2/docs/tutorials/rag/
