import streamlit as st 
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

# Lets Write the Code to Load the File
def load_doc(file):
    name,extention = os.path.splitext(file)
    if extention == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file)
        data = loader.load()
        return data
    elif extention == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file)
        data = loader.load()
        return data
    elif extention == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
        data = loader.load()
        return data
    return None


# Write the Functions to Divide the data into Chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


def delete_pinecone_index(index_name='all'):
    import pinecone
    pc = pinecone.Pinecone()
    
    if index_name == 'all':
        indexes = pc.list_indexes().names()
        st.write('Deleting all indexes ... ')
        for index in indexes:
            pc.delete_index(index)
        st.write('Ok')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pc.delete_index(index_name)
        print('Ok')
    
def insert_or_fetch_embeddings(chunks, index_name='askadocument'):
    # Importing necessary libraries
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from sentence_transformers import SentenceTransformer
    from pinecone import ServerlessSpec
    import streamlit as st

    # Initialize Pinecone client
    pc = pinecone.Pinecone()
    
    # Load the local sentence-transformer model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    local_model = SentenceTransformer(model_name)

    # Define a function to generate embeddings
    def generate_embeddings(documents):
        return local_model.encode(documents).tolist()  # Convert to list for compatibility

    # Loading from existing index
    if index_name in pc.list_indexes().names():
        st.write(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name, generate_embeddings)
        st.write('Ok')
    else:
        # Creating the index and embedding the chunks into the index 
        st.write(f'Creating index {index_name} and embeddings ...')

        # Creating a new index
        pc.create_index(
            name=index_name,
            dimension=384,  # MiniLM-L6-v2 produces 384-dimensional embeddings
            metric='cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        # Processing the input documents, generating embeddings, and inserting into the index
        vector_store = Pinecone.from_documents(chunks, generate_embeddings, index_name=index_name)
        st.write('Ok')
        
    return vector_store

def fetch_embeddings(index_name = 'askadocument'):
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from sentence_transformers import SentenceTransformer
    pc = pinecone.Pinecone()
    
    # Load the local sentence-transformer model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    local_model = SentenceTransformer(model_name)

    # Define a function to generate embeddings
    def generate_embeddings(documents):
        return local_model.encode(documents).tolist()
    if index_name in pc.list_indexes().names():
        st.write(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name, generate_embeddings)
        return vector_store


# Now we need to calculate the cost before uploading it to vectore database
# to avoid the excess priceing

def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # check prices here: https://openai.com/pricing
    return total_tokens,round(total_tokens / 1000 * 0.00002,6)

# Now lets write the main code for LLM
# to ask the question 

# 
def ask_and_get_answer(vector_store, q, k=3):
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains import RetrievalQA

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    # Create the RetrievalQA chain with the LLM and the retriever
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    # Invoke the chain with the question
    answer = chain.invoke(q)
    return answer

# Now lets Write the Code for Streamlit to create the web app

if __name__ == '__main__':
    # 1) load Api in environment for security
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True) 

    # 2) Add Img to Make UI More Friendly
    st.image('image3.png')

    st.subheader('LLM Question-Answering Application 🤖')
    st.subheader('Insightful Answers 🌐')

    # 3) Now we crete the Sidebar to our web

    with st.sidebar:
        # User can also load the api
        api_key = st.text_input('OpenAI API_Key: ',type='password')
        if api_key:
            os.environ['GOOGLE_API_KEY'] = api_key

        # Lets give acces to upload the file
        upload_file = st.file_uploader('UPload the File: ',type=['pdf','docx','txt'])

        # Also Provide the accsees the user to choose 
        # Chunk Size and k_value
        chunk_size = st.number_input('Chunk Size: ',max_value=2048,min_value=100,value=512)
        k = st.number_input('K',max_value=10,min_value=1,value=3)

        add_data = st.button('Add Data')

        if upload_file and add_data:
            with st.spinner('Reading , Chunking and Embedding File'):
                # Writing the file from RAM to the current directory
                bytes_data = upload_file.read()

                file_name = os.path.join('./',upload_file.name)
                with open(file_name,'wb') as f:
                    f.write(bytes_data)

                # now load the data to function
                data = load_doc(file_name)

                # lets Convert to chunks and we take input from use chunk size in above
                chunks = chunk_data(data,chunk_size=chunk_size)

                st.write(f'Chunk Size: {chunk_size}, Chunks: {len(chunks)}')

                # Show the total cost of embeddings
                tokens,embedding_cost = calculate_embedding_cost(chunks)

                st.write(f'embedding price: {embedding_cost}')

                # Creating the embeddings and returning the PINECONE vextore store
                delete_pinecone_index()
                vectore_store = insert_or_fetch_embeddings(chunks=chunks)
                if vectore_store != None:
                    st.write('Vectore Exist')

                # saving the vectore store to the streamlit session to 
                # persistant between rerun
                st.session_state.vs = vectore_store
                st.success('File Uploaded, Chunked and embedded successfully')
# Now take question as input
q = st.text_input('Ask a Question on document: ')
if q:
    standard_answer = '''Answer only based on the text you received as input. Don't search external sources. " \
                        "If you can't answer then return `I DONT KNOW`.'''
    q = f"{q} {standard_answer}"

    if 'vs' in st.session_state:
        vectore_store = fetch_embeddings()
        answer = ask_and_get_answer(vectore_store,q,k=k)
        # show Ans
        st.text_area('LLM Answer: ',value=answer['result'],height=400)
    else:
        st.write("First Upload the Document")
