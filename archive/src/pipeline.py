import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Pydantic Schema
class RiskAssessment(BaseModel):
    risk_category: str = Field(description="The category of risk")
    severity_score: int = Field(description="Risk severity on a scale from 1 to 10")
    reasoning: str = Field(description="Explanation of the score")
    evidence: str = Field(description="Exact quote from the text")

def setup_rag_and_analyze():
    print("1. Loading Data & Embedding Model...")
    sample_text = """
    Item 1A. Risk Factors.
    Liquidity Risk: As of December 31, 2024, the Company had $1.2 billion in cash and cash equivalents. 
    However, our short-term obligations over the next 12 months total $1.5 billion. If we are unable to 
    secure additional financing, we may face a liquidity crisis.
    Litigation Risk: The Company is currently facing a class-action lawsuit regarding environmental 
    compliance at our primary manufacturing facility. Potential damages could exceed $500 million.
    """
    
    with open("temp_sample.txt", "w") as f:
        f.write(sample_text)

    loader = TextLoader("temp_sample.txt")
    docs = loader.load()
    
    # Using the fixed chunk size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db_pipeline")

    print("2. Retrieving context for: 'environmental litigation'")
    # query for the other risk in our text this time
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    retrieved_docs = retriever.invoke("What are the environmental litigation risks?")
    context_string = retrieved_docs[0].page_content

    print("3. Passing context to Gemini Agent...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(RiskAssessment)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert financial analyst. Extract risk information from the provided SEC 10-K filing excerpt. You must output the data matching the requested schema."),
        ("human", "Filing Excerpt:\n{text}")
    ])
    
    chain = prompt | structured_llm
    
    # feed the RAG output directly to the Agent input
    result = chain.invoke({"text": context_string})
    
    print("\n=== FINAL PIPELINE OUTPUT ===")
    print(result.model_dump_json(indent=2))

    if os.path.exists("temp_sample.txt"):
        os.remove("temp_sample.txt")

if __name__ == "__main__":
    setup_rag_and_analyze()