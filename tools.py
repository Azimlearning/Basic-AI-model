from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import ddg_search
from langchain.tools import Tool
from datetime import datetime

'''
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.tools import GoogleSearchRun
google_search_api_wrapper = GoogleSearchAPIWrapper()
google_search_tool = GoogleSearchRun(api_wrapper=google_search_api_wrapper)
'''


# Create API wrappers
# wikipedia_api_wrapper = WikipediaAPIWrapper()
duck_duck_go_search_tool = ddg_search.DuckDuckGoSearchRun()

'''
# Pass wrapper into tool
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)

# Example: Your unified search tool
search_tool = Tool.from_function(
    func=duck_duck_go_search_tool.run,
    name="DuckDuckGo Search",
    description="Search the web using DuckDuckGo"
)
'''

search = ddg_search.DuckDuckGoSearchRun()
search_tool = Tool(
    func=search.run,
    name="DuckDuckGo_Search",
    description="Search the web using DuckDuckGo"
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

save_tool = Tool(
    func=save_to_txt,
    name="Save_to_Text_File",
    description="Saves the research output to a text file"
)
