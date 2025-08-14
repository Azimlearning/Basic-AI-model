from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables import RunnableSequence
from tools import search_tool, wikipedia_tool, save_tool



load_dotenv()  # Loads API keys from .env

class Responce(BaseModel):
    topic : str
    summary : str
    sources : list[str]
    tools_used : list[str] 


llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="z-ai/glm-4-32b", 
    #openai/gpt-oss-20b:free
    #openai/gpt-5-nano to small token
    #openai/gpt-oss-120b
    #z-ai/glm-4.5-air:free rate limit
    #z-ai/glm-4-32b
    temperature=0,
    max_completion_tokens=2000,
    model_kwargs={"response_format": {"type": "json_object"}} 
)

parser = PydanticOutputParser(pydantic_object=Responce)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", """
You are a chef assistant that will help generate a recipe.

Return your answer only in valid JSON, matching this schema:
{format_instructions}

For `tools_used`, list the names of the AI tools you used during the search and summarization process, not cooking utensils.

Do not return the schema itself.
Fill in the values for each field.
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

#Wrap your response strictly in JSON matching this schema:\n{format_instructions}


#tools = [search_tool, wikipedia_tool, save_tool] #

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools= []
)

agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True) #

quary = input("Enter your query: ")
raw_response = agent_executor.invoke({
    "query": quary,
    "chat_history": [],
    "agent_scratchpad": []
})

# Parse directly
chain = prompt | llm | parser

''''''
#print testing
print("\n--- RAW RESPONSE ---")
print(raw_response)

# Show only the output field
print("\n--- OUTPUT FIELD ---")
print(raw_response.get("output"))

# Parse to Pydantic model
try:
    structured_response = parser.parse(raw_response["output"])
    print("\n--- PARSED STRUCTURED RESPONSE ---")
    print(structured_response)  # Pretty print
    print("\nTopic:", structured_response.topic)
    print("Summary:", structured_response.summary)
    print("Sources:", structured_response.sources)
    print("Tools Used:", structured_response.tools_used)
except Exception as e:
    print("Error parsing response:", e)
