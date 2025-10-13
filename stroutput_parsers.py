from langchain_OpenAI import OpenAI 
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
llm=OpenAI(temperature=0)
template=promptTemplate(
    template="tell me a joke about {subject}",input_variables=["subject"]
    
)
prompt=template.invoke({"subject":"chickens"})
result=llm.invoke(prompt)
template1=promptTemplate(template="tell me summary of this text in 5 lines {text}",input_variables=["text"]
                         )
parser=StrOutputParser()
chain=template|llm|parser|template1|llm|parser
