import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load API keys from the .env file
load_dotenv()

def run_basic_rag():
    print("1. Initializing Embedding Model...")
    # text-embedding-3-large as specified in project proposal
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    print("2. Creating sample financial data...")
    # use a hardcoded string just to test the pipeline. 
    # swap this out for the FinanceBench dataset.
    sample_10k_text = """
    Item 1A. Risk Factors.
    Liquidity Risk: As of December 31, 2024, the Company had $1.2 billion in cash and cash equivalents. 
    However, our short-term obligations over the next 12 months total $1.5 billion. If we are unable to 
    secure additional financing, we may face a liquidity crisis.
    Litigation Risk: The Company is currently facing a class-action lawsuit regarding environmental 
    compliance at our primary manufacturing facility. Potential damages could exceed $500 million.
    """
    
    # Save the sample to a temporary file so LangChain can load it
    with open("temp_sample.txt", "w") as f:
        f.write(sample_10k_text)

    print("3. Loading and Chunking Text...")
    loader = TextLoader("temp_sample.txt")
    docs = loader.load()
    
    # How to split the text changes what the AI "sees"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    print(f"   -> Split text into {len(splits)} chunks.")

    print("4. Storing in ChromaDB...")
    # create a local database folder in your project
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")

    print("5. Running a Retrieval Query...")
    query = "What is the company's liquidity situation?"
    print(f"\nQuery: {query}")
    
    # Retrieve the top 2 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    results = retriever.invoke(query)
    
    print("\n--- Retrieved Context ---")
    for i, res in enumerate(results):
        print(f"\nResult {i+1}:")
        print(res.page_content)

    # Clean up the temp file
    if os.path.exists("temp_sample.txt"):
        os.remove("temp_sample.txt")

if __name__ == "__main__":
    run_basic_rag()