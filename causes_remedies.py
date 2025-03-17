from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import os
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are helpful assistant, Please Provide response to the user queries"),
        ("user", "Question:Causes and Remedies for {disease} disease"),
    ]
)

llm = Ollama(model="llama3.2")
output_parse = StrOutputParser()

chain = prompt|llm|output_parse

def call_causes(disease):
    return chain.invoke({'disease': disease})