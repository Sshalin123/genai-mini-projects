import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import openai
from dotenv import load_dotenv

load_dotenv()
os.environ['openai_api']=os.getenv("openai_api")
os.environ['groq']=os.getenv("groq")
groq_api_key=os.getenv("groq")

# chat template 
llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")
prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}

   """
)

# FRONTEND WITH STREAMLIT   
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OpenAIEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("research_papers") ## S1 Data Ingestion 
        st.session_state.docs=st.session_state.loader.load() ## S2 Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## S3 Text Splitting
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50]) # S4 Document Splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) # S5 Vector Store Creation
st.title("RAG Document Q&A With Groq And Lama3")
user_prompt=st.text_input("Enter your query from the research paper")
if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

""" 
RAG Model for DOCUMENT Q&A . 
calculkating time of response and difference betweeen use of open ai api key and groq api key 
just bcz open ai is paid and groq is open source so that we can get a better idea to decide which is better 
"""
import time

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt})
    print(f"Response time :{time.process_time()-start}")
    st.write(response['answer'])
    with st.expander("Document similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')






