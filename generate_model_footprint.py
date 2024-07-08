import os
import base64




from operator import itemgetter
from langchain.schema import format_document

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string, BaseMessage
from langchain.memory import ConversationBufferMemory

from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from pprint import pprint


def get_system_message():
    with open("prompts/system_message_for_model_generator.txt", "r") as f:
        system_message = f.read()
    return system_message

def get_human_message():
    with open("prompts/human_message_for_model_generator.txt", "r") as f:
        human_message = f.read()
    return human_message


def get_prompt(service: str) -> list[BaseMessage]:
    system_message = get_system_message()
    human_message = get_human_message().format(service=service)


    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content=(
                system_message
            )
        ),
        HumanMessage(
            content=[
                {"type": "text", 
                #"text": f"For the Azure service, named: {service}, provide a detailed model / formula of how to compute an estimate of the carbon cloud footprint leveraging the knowledge about the service. Make required assumptions for the input parameters of the model and then provide an estimate of the carbon cloud footprint of the component. Ensure you clearly describe the model parameters, the assumptions on their values, the formula(s), and the computation of the estimate using the formula(s). The parameters of the model / formula must include datacenter PUE and carbon intensity among others. The output must be organized in only and only the following 'Description of model', 'Model Parameters', 'Assumptions', 'Formulas', 'Computation', and 'Conclusion'. The output must be in json format to allow for digital processing",
                "text": human_message,
                },
            ]
        )
    ])
    return prompt

def get_model_footprint(components):
    llm = ChatOpenAI(temperature=0.3, model="gpt-4-turbo-preview", max_tokens=3000, )
    chain = RunnableLambda(get_prompt) |  llm

    outputs = []
    for component in components:
        print(component)
        output = chain.invoke({'service':component})
        outputs.append(output)
    return outputs




