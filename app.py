import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import google.generativeai as genai
import os

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load and vectorize the website content
def get_vectorstore_from_url(url):
    # Load the website content
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # Split the content into chunks for vectorization
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    document_chunks = text_splitter.split_documents(document)

    # Ensure chunks are valid strings for vectorization
    document_chunks = [chunk.page_content for chunk in document_chunks if chunk.page_content]

    # If no valid chunks are found, return an error
    if not document_chunks:
        st.error("All extracted chunks are empty or invalid.")
        return None
    
    # Use Google Generative AI for embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create FAISS vector store from document chunks
    vector_store = FAISS.from_texts(document_chunks, embeddings)
    
    return vector_store

# Function to create a conversational chain using the Gemini model
def get_conversational_chain():
    # Define a prompt template for the conversation
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available in the context, say,
    'The answer is not available in the context'. Do not provide an incorrect answer.

    Context: {context}
    Question: {question}

    Answer:
    """
    
    # Initialize the Gemini model with specific parameters
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    
    # Set up the prompt and chain using the Gemini model
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

# Function to get the response from the conversational chain
def get_response(user_input, vector_store):
    # Search for relevant documents in the vector store
    docs = vector_store.similarity_search(user_input)
    
    # Create the conversational chain using Gemini
    chain = get_conversational_chain()
    
    # Run the chain and get the response
    response = chain({"input_documents": docs, "question": user_input}, return_only_outputs=True)
    
    return response["output_text"]

# Streamlit app configuration
st.set_page_config(page_title="Chat with Websites", page_icon="🤖")
st.title("Chat with Websites using Gemini")

# Sidebar for website URL input
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Enter Website URL")
    if st.button("Process"):
        if website_url:
            with st.spinner(""):
                st.session_state.vector_store = get_vectorstore_from_url(website_url)
                st.session_state.vector_store_created = True
        else:
            st.error("Please enter a valid URL")

# Check if the vector store has been created
if "vector_store_created" in st.session_state and st.session_state.vector_store_created:
    # Initialize chat history if it's not already present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I assist you?")
        ]
    
    # Input box for user query
    user_query = st.chat_input("Ask a question based on the website content...")

    if user_query:
        # Get response from Gemini conversational chain
        response = get_response(user_query, st.session_state.vector_store)
        
        # Update chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Display chat history in the app
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
