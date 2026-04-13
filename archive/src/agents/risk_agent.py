import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load your API keys from the .env file
load_dotenv()

# 1. Define the Expected JSON Structure using Pydantic
class RiskAssessment(BaseModel):
    risk_category: str = Field(description="The category of risk (e.g., Liquidity, Litigation, Operational, Market)")
    severity_score: int = Field(description="Risk severity on a scale from 1 to 10")
    reasoning: str = Field(description="A brief explanation of why this score was given based on the text")
    evidence: str = Field(description="An exact quote from the text that proves the risk")

def run_risk_agent():
    print("1. Initializing the Gemini LLM...")
    # We use gemini-2.5-flash for structured extraction
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # 2. Bind the Pydantic schema to force JSON output
    structured_llm = llm.with_structured_output(RiskAssessment)
    
    # 3. Create a System Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert financial analyst. Extract risk information from the provided SEC 10-K filing excerpt. You must output the data matching the requested schema."),
        ("human", "Filing Excerpt:\n{text}")
    ])
    
    # 4. Create the Chain
    chain = prompt | structured_llm
    
    # 5. retrieved chunk from our earlier RAG experiment
    retrieved_chunk = """
    Item 1A. Risk Factors.
    Liquidity Risk: As of December 31, 2024, the Company had $1.2 billion in cash and cash equivalents. 
    However, our short-term obligations over the next 12 months total $1.5 billion. If we are unable to 
    secure additional financing, we may face a liquidity crisis.
    """
    
    print("2. Analyzing risk and generating structured JSON...")
    result = chain.invoke({"text": retrieved_chunk})
    
    print("\n--- Agent Output (Python Object) ---")
    print(f"Category: {result.risk_category}")
    print(f"Severity: {result.severity_score}/10")
    print(f"Reasoning: {result.reasoning}")
    print(f"Evidence: {result.evidence}")
    
    print("\n--- Raw JSON Output ---")
    print(result.model_dump_json(indent=2))

if __name__ == "__main__":
    run_risk_agent()