from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import pdfplumber
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

llm = ChatOpenAI(model="gpt-4o-mini")

# AI-powered PDF analysis for spending habits
def analyze_spending_from_pdf(pdf_path: str):
    with pdfplumber.open(pdf_path) as pdf:
        raw_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

    if not raw_text:
        return "No text extracted from PDF. Ensure the PDF is not an image-based scan."

    prompt = f"""
    You are a financial assistant analyzing a Wells Fargo bank statement.
    Extract transactions, categorize them, and provide insights including:

    - Total spending for the month
    - Amount spent by category (Dining, Groceries, Bills, etc.)
    - Unusual transactions (large or unexpected expenses)
    - Budget improvement suggestions and opinion on spending habits.

    Here is the statement text:
    {raw_text}
    """

    response = llm.invoke(prompt)
    return response

# Define LangChain tool
ai_spending_tool = Tool(
    name="analyze_spending_from_pdf",
    func=analyze_spending_from_pdf,
    description="Extracts and analyzes spending habits from a Wells Fargo PDF statement."
)