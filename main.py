from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import ai_spending_tool
import os
import json

load_dotenv()

class ResearchResponse(BaseModel):
    total_spending: str
    categories: str
    unusual_transactions: list[str]
    suggestions: str
    tools_used: list[str]

llm = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a finance assistant that analyzes Wells Fargo statements and provides spending insights.\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [ai_spending_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("Enter the filename of your Wells Fargo PDF statement: ").strip()
pdf_path = os.path.join(os.getcwd(), query)
raw_response = agent_executor.invoke({"query": query})
# print(raw_response)

try:
    structured_response = parser.parse(raw_response.get("output"))
    print(json.dumps(structured_response.model_dump(), indent=4))
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)