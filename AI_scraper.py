import streamlit as st
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """
You are an assistant for question-answering tasks> Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Don't try to make up an answer. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

embeddings = OllamaEmbeddings(model="llama3.2")
vectorstore = InMemoryVectorStore(embeddings)

model = OllamaLLM(model="llama3.2")

def load_page(url):
    loader = SeleniumURLLoader(
        urls=[url]
    )
    documents = loader.load()
    return documents

def split_page(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    data = text_splitter.split_documents(documents)

    return data

def index_docs(documents):
    vectorstore.add_documents(documents)

def retrieve_docs(query):
    return vectorstore.similarity_search(query)

def answer_question(question, context):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})


st.title("AI Crawler")
url = st.text_input("Enter the URL of the page you want to scrape:")

if url:
    documents = load_page(url)
    chunked_docs = split_page(documents)

    index_docs(chunked_docs)
    st.write("Page loaded and indexed.")
    question = st.chat_input("Ask a question about the page:")

    if question:
        st.chat_message("user").write(question)  # Display the user's question in the Streamlit app

        retrieve_docs = retrieve_docs(question)
        context = "\n\n".join([doc.page_content for doc in retrieve_docs])
        answer = answer_question(question, context)
        st.chat_message("assistant").write(answer)  # Display the assistant's answer in the Streamlit app

    