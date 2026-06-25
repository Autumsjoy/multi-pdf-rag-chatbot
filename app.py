from dotenv import load_dotenv
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css,bot_template, user_template

custom_template="""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original Language.
Chat History:{chat_history}
Follow up input :{question}
Standalone question:
"""
Custom_question_prompt= PromptTemplate.from_template(custom_template)
 #extracting the texts from pdf
def get_pdf_text(docs):
    text=""
    for pdf in docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
            
    return text
def get_chunks(raw_text):
    text_splitter=CharacterTextSplitter(separator="\n",
                                        chunk_size=1000,
                                        chunk_overlap=300,
                                        length_function=len)
    chunks= text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(chunks):
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                     model_kwargs={'device':'cpu'})
    vectorstore=FAISS.from_texts(texts=chunks,embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    groq_key=os.getenv("GROQ_API_KEY")    
    llm = ChatOpenAI( base_url="https://api.groq.com/openai/v1",model="llama-3.1-8b-instant",
        api_key=groq_key,
        temperature=0.1)
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=Custom_question_prompt,
        memory=memory
    )
    return conversational_chain

def handle_question(question):
    response=st.session_state.conversation({'question':question})
    st.session_state.chat_history=response["chat_history"]
    for i, msg in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}",msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",msg.content),unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css,unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    if  "chat_history" not in st.session_state:
        st.session_state.chat_history=None
    st.header("Chat with multiple PDFS :books:") 
    question=st.text_input("Ask question from your document:")
    if question:
        handle_question(question)
    with st.sidebar:
        st.subheader("Your Documents")
        doc=st.file_uploader("Upload your pdf here and click on 'Process' ",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                 raw_text=get_pdf_text(doc)
                 text_chunks=get_chunks(raw_text)
                 vectorstore=get_vectorstore(text_chunks)
                 st.session_state.conversation=get_conversation_chain(vectorstore)
                 
if __name__=='__main__':
    main()