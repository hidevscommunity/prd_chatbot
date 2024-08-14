import json

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import HuggingFaceEmbeddings  # General embeddings from HuggingFace models.
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub, LlamaCpp, CTransformers  # For loading transformer models.
from langchain.document_loaders import PyPDFLoader, TextLoader, JSONLoader, CSVLoader
import tempfile  # Library for creating temporary files.
import os

def export_chat_history():
    chat_history = st.session_state.chat_history
    if chat_history:
        file_path = st.file_uploader("Choose where to save the chat history", type="json", accept_multiple_files=False)
        if file_path:
            file_path = file_path.name
            with open(file_path, "w") as f:
                json.dump(chat_history, f, default=lambda x: x.__dict__)
            st.success(f"Chat history exported to {file_path}")
    else:
        st.warning("No chat history available to export")
# Function to extract text from PDF documents.
def get_pdf_text(pdf_docs):
    temp_dir = tempfile.TemporaryDirectory()  # Create a temporary directory.
    temp_filepath = os.path.join(temp_dir.name, pdf_docs.name)  # Create a temporary file path.
    with open(temp_filepath, "wb") as f:  # Open the temporary file in binary write mode.
        f.write(pdf_docs.getvalue())  # Write the content of the PDF document to the temporary file.
    pdf_loader = PyPDFLoader(temp_filepath)  # Load the PDF using PyPDFLoader.
    pdf_doc = pdf_loader.load()  # Extract text.
    return pdf_doc  # Return the extracted text.


# Task
# Write the text extraction function below.

def get_text_file(docs):
    temp_dir = tempfile.TemporaryDirectory()  # Create a temporary directory.
    temp_filepath = os.path.join(temp_dir.name, docs.name)  # Create a temporary file path.
    with open(temp_filepath, "wb") as f:  # Open the temporary file in binary write mode.
        f.write(docs.getvalue())
    loader = TextLoader(temp_filepath)
    data = loader.load()
    return data


def get_csv_file(docs):
    temp_dir = tempfile.TemporaryDirectory()  # Create a temporary directory.
    temp_filepath = os.path.join(temp_dir.name, docs.name)  # Create a temporary file path.
    with open(temp_filepath, "wb") as f:  # Open the temporary file in binary write mode.
        f.write(docs.getvalue())
    loader = CSVLoader(file_path=temp_filepath)
    data = loader.load()
    return data


def get_json_file(docs):
    temp_dir = tempfile.TemporaryDirectory()  # Create a temporary directory.
    temp_filepath = os.path.join(temp_dir.name, docs.name)  # Create a temporary file path.
    with open(temp_filepath, "wb") as f:  # Open the temporary file in binary write mode.
        f.write(docs.getvalue())
    loader = JSONLoader(file_path=temp_filepath,
                        jq_schema='.messages[].content',
                        text_content=False)
    data = loader.load()

    return data


# Function to split documents into text chunks.
def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Specify the chunk size.
        chunk_overlap=200,  # Specify the overlap between chunks.
        length_function=len  # Specify the function for measuring text length.
    )

    documents = text_splitter.split_documents(documents)  # Split documents into chunks.
    return documents  # Return the chunks.


# Function to create a vector store from text chunks.
def get_vectorstore(text_chunks):
    # Load the OpenAI embedding model. (Embedding models - Ada v2)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(text_chunks, embeddings)  # Create a FAISS vector store.

    return vectorstore  # Return the created vector store.


def get_conversation_chain(vectorstore):
  """
  This function creates a conversational retrieval chain for analyzing product requirements documents.

  Args:
      vectorstore: A vector store object used for retrieval.

  Returns:
      A ConversationalRetrievalChain object.
  """

  # Use a more descriptive name for the gpt model
  gpt_model_name = 'gpt-3.5-turbo'
  gpt_model = ChatOpenAI(model_name=gpt_model_name)  # Load the gpt-3.5 model

  from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

  # Create system message template (consider making this an argument)
  system_template = """
    You are an AI assistant specialized in generating test cases for software features. Your task is to create test cases based on the list of features provided. Each test case should include:
    - A title
    - Description of what is being tested
    - Preconditions
    - Test steps
    - Expected results

    Instructions:

    1. Read the list of features carefully.
    2. Generate Test Cases: For each identified feature, create one or more test cases including:
        - Title
        - Description
        - Preconditions
        - Test Steps
        - Expected Results

    Output Format:

    Provide the test cases and Python code in the following format:

    1. Title: Brief title of the test case
        - Description: Detailed description of the test case.
        - Preconditions: List of preconditions.
        - Test Steps: Step-by-step instructions.
        - Expected Results: Expected outcome.

            {context}"""

  # Create chat prompt templates
  question_prompt = "{question}"  # Consider making this an argument
  messages = [
      SystemMessagePromptTemplate.from_template(system_template),
      HumanMessagePromptTemplate.from_template(question_prompt)
  ]
  question_answer_prompt = ChatPromptTemplate.from_messages(messages)

  # Create memory to store conversation history.
  memory = ConversationBufferMemory(
      memory_key='chat_history', return_messages=True)

  # Handle potential errors with try-except block
  try:
      # Create a conversational retrieval chain.
      conversation_chain = ConversationalRetrievalChain.from_llm(
          llm=gpt_model,
          retriever=vectorstore.as_retriever(),
          memory=memory,
          combine_docs_chain_kwargs={"prompt": question_answer_prompt},
      )
      return conversation_chain
  except Exception as e:
      print(f"Error creating conversation chain: {e}")
      return None




# Function to handle user input.
def handle_userinput(user_question):
    # Generate a response to the user question using the conversation chain.
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    # Store the conversation history.

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)



def main():
    load_dotenv()
    st.header("PRD Test Cases :")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
        
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    

    with st.sidebar:
        openai_key = st.text_input("Paste your OpenAI API key (sk-...)")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key

        st.subheader("Your documents")
        docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                doc_list = []

                for file in docs:
                    print('file - type : ', file.type)
                    if file.type == 'text/plain':
                        # file is .txt
                        doc_list.extend(get_text_file(file))
                    elif file.type in ['application/octet-stream', 'application/pdf']:
                        # file is .pdf
                        doc_list.extend(get_pdf_text(file))
                    elif file.type == 'text/csv':
                        # file is .csv
                        doc_list.extend(get_csv_file(file))
                    elif file.type == 'application/json':
                        # file is .json
                        doc_list.extend(get_json_file(file))

                # get the text chunks
                text_chunks = get_text_chunks(doc_list)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                


    # Export chat history button
    if st.button("Export Chat History"):
        export_path = "test.json"
        if export_path:
            chat_history = st.session_state.chat_history
            if chat_history:
                with open(export_path, "w") as f:
                    json.dump(chat_history, f, default=lambda x: x.__dict__)
                st.success("Chat history exported successfully.")
            else:
                st.warning("No chat history available to export.")

if __name__ == '__main__':
    main()
