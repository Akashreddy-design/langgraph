from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model1 = ChatOpenAI(temperature=0)
parser1 = StrOutputParser()

class Temp(BaseModel):
    choice: Literal["positive","negative"] = Field(
        description="choose positive or negative based on the sentiment of the review"
    )

parser = PydanticOutputParser(pydantic_object=Temp)

template1 = PromptTemplate(
    template="tell me sentiment of the review text {text},\n{format_instruction}",
    input_variables=["text"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)
template2 = PromptTemplate(
    template="give a response for the positive feedback {feedback}",
    input_variables=["feedback"]
)
template3 = PromptTemplate(
    template="give a response for the negative feedback {feedback}",
    input_variables=["feedback"]
)

# Sentiment detection chain
chain = template1 | model1 | parser

# Branching chain
branch_chain = RunnableBranch(
    (lambda x: x.choice=="positive", template2 | model1 | parser1),
    (lambda x: x.choice=="negative", template3 | model1 | parser1),
    RunnableLambda(lambda x:  "invalid sentiment")
)

# Final chain
final_chain = chain | branch_chain

result = final_chain.invoke({"text":"this is terrible movie"})
print(final_chain.get_graph().print_ascii())
