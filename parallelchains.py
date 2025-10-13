from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser   
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from pydantic import BaseModel, Field

load_dotenv()
model1=ChatOpenAI(temperature=0)
model2=ChatOpenAI(temperature=0)


template1=PromptTemplate(
    template="tell me detailed description about {text}",input_variables=["text"]
)
template2=PromptTemplate(
    template="give me question and answer about this text {text}",input_variables=["text"]
)
template3=PromptTemplate(
    template="geneate a combined summary of these two texts {text1} and {text2}",input_variables=["text1","text2"]
)   
parser=StrOutputParser()
parallel_chain=RunnableParallel(
    {
        "text1":template1|model1|parser,
        "text2":template2|model2|parser    
    }
)
last_chain=template3|model1|parser
final_chain=parallel_chain|last_chain
result=final_chain.invoke({"text":"blackholes"})
print(result) 
final_chain.get_graph().print_ascii()
