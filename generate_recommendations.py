import os
import base64

from dotenv import load_dotenv


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



# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_recommendations(image_path):
    
    with open("./prompts/system_message_generate_recommendations_v2.txt", "r") as f:
        system_message = f.read()
    

    with open("./prompts/human_message_generate_recommendations_v2.txt", "r") as f:
        human_message = f.read()
    

    # Getting the base64 string
    base64_image = encode_image(image_path)

    messages = [
        SystemMessage(
            content=(
                {system_message}
            )
        ),
        HumanMessage(
            content=[
                {"type": "text", 
                "text": f"""{human_message}"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "auto",
                    },
                },
            ]
        )]
    #print("hello")
    #print(messages)
    #exit()
    llm = ChatOpenAI(temperature=0.3, model="gpt-4-vision-preview", max_tokens=3000, )
    chain =  ChatPromptTemplate.from_messages(messages)  | llm

    #print(system_message)
    #print(human_message)
    #print(base64_image)
    #response = chain.invoke({"system_message": system_message, "human_message": human_message, "base64_image": base64_image})
    response = chain.invoke({})
    
    def getSubstringBetweenTwoChars(ch1,ch2,s):
        return s[s.find(ch1)+1:s.find(ch2)]
    x= getSubstringBetweenTwoChars('[', ']', response.content).replace("\n", "")
    components=[]
    for component in x.split(","):
        components.append(component.strip().replace("\"", ""))
    return components


#x=load_dotenv()
#image_path="C:/Users/MANISHGUPTA/kyndryl/github/solution-sustainability/images/architecture.png"
#generate_recommendations(image_path)