## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
st.title("Conversational RAG With PDF uplaods and chat history")
st.write("Upload Pdf's and chat with their content")
api_key=st.text_input("Enter your Groq API key:",type="password")
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")
    session_id=st.text_input("Session ID",value="default_session")
    if 'store' not in st.session_state:
        st.session_state.store={}
    uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()    
        
        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)
        system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
        qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
      
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        user_input = st.text_input("whats in your mind ?")
        if user_input:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },  # constructs a key "abc123" in `store`.
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the GRoq API Key")

# Explanation of the Conversational RAG with PDF Uploads and Chat History Code for newbies
Key Components
1. Initial Setup
Imports necessary libraries including LangChain components, Streamlit, and PDF processing tools
Loads environment variables (like HuggingFace token)
Sets up HuggingFace embeddings model (all-MiniLM-L6-v2)

2. Streamlit Interface
Creates a title and description for the web app
Provides an input field for the Groq API key (required to use the LLM)
Only proceeds if an API key is provided

3. Chat System Initialization
Creates a ChatGroq instance with the Gemma2-9b-It model
Sets up a session ID system to maintain separate chat histories
Initializes a session state store to keep track of conversations

4. PDF Processing
Provides a file uploader for multiple PDFs
For each uploaded PDF:
Saves it temporarily
Uses PyPDFLoader to extract text
Splits text into chunks using RecursiveCharacterTextSplitter
Creates embeddings and stores them in a Chroma vector database

5. RAG (Retrieval-Augmented Generation) Setup
Contextualization Prompt: Reformulates questions considering chat history
History-Aware Retriever: Finds relevant documents considering conversation context
QA System Prompt: Instructions for how to answer questions
QA Chain: Combines retrieval and question answering

6. Conversation Management
get_session_history function maintains separate chat histories per session
conversational_rag_chain combines everything with message history handling

7. User Interaction
Text input for user questions
When a question is submitted:
Retrieves the session's chat history
Invokes the RAG chain with the question and history
Displays the assistant's response
Shows the current chat history and session state

Workflow
User enters Groq API key
User uploads PDF files
System processes and indexes the PDF content
User asks questions about the PDF content

System:
Considers chat history to understand context
Retrieves relevant information from PDFs
Generates answers using the LLM
Maintains conversation history for future questions

Key Features
Persistent Chat History: Maintains context across multiple questions
Multi-PDF Support: Can process and query across multiple uploaded documents
Session Isolation: Different users/sessions get separate chat histories
Conversational Context: Understands follow-up questions by referencing history
The application is particularly useful for document-based question answering where the conversation might span multiple related questions.







