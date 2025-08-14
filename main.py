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
import traceback

load_dotenv()  # Loads API keys from .env

class Responce(BaseModel):
    topic : str
    summary : str
    ingredients : list[str]
    instructions : list[str]
    sources : list[str]
    tools_used : list[str] 

def create_llm(max_tokens=None, llm_model=str):
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=llm_model,
        temperature=0,
        max_completion_tokens=max_tokens,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

llm = create_llm(max_tokens=None, llm_model="z-ai/glm-4-32b")   
llm_2 = create_llm(max_tokens=None, llm_model="openai/gpt-oss-20b:free")
llm_3 = create_llm(max_tokens=None, llm_model="openai/gpt-5-nano")          #to small token
llm_4 = create_llm(max_tokens=None, llm_model="openai/gpt-oss-120b")
llm_5 = create_llm(max_tokens=None, llm_model="z-ai/glm-4.5-air:free")      #rate limit

llm_fallback = create_llm(max_tokens=1000, llm_model="z-ai/glm-4-32b")


backup_llms = [llm_2, llm_3, llm_4, llm_5, llm]

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

def run_with_token_fallback(query):
    global llm, agent_executor

    try:
        return agent_executor.invoke({
            "query": query,
            "chat_history": [],
            "agent_scratchpad": []
        })

    except Exception as e:
        err_msg = str(e).lower()
        print("\n[!] Error encountered:", e)

        # Credit/token handling
        if (
            "max tokens" in err_msg
            or "token limit" in err_msg
            or "too many tokens" in err_msg
            or "requires more credits" in err_msg
        ):
            print("[!] Token/Credit issue — retrying with reduced max_completion_tokens.:")

            agent_fallback = create_tool_calling_agent(
                llm=llm_fallback,
                prompt=prompt,
                tools=[]
            )
            executor_fallback = AgentExecutor(agent=agent_fallback, tools=[], verbose=True)

            return executor_fallback.invoke({
                "query": query,
                "chat_history": [],
                "agent_scratchpad": []
            })

        else:
            raise

# ✅ Use fallback runner here
print("Hello i am you personal chef")
query = input("what recipe would you like to cook?: ")
raw_response = run_with_token_fallback(query)

'''
query = input("Enter your query: ")
raw_response = agent_executor.invoke({
    "query": query,
    "chat_history": [],
    "agent_scratchpad": []
})
'''

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
    print("\nSummary:", structured_response.summary)
    print("\nIngredients:", structured_response.ingredients)
    print("\nInstructions:", structured_response.instructions)
    print("\nSources:", structured_response.sources)
    print("\nTools Used:", structured_response.tools_used)
except Exception as e:
    print("Error parsing response:", e)
