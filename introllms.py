import langchain
from langchain_openai import ChatOpenAI
import os # https://docs.python.org/3/library/os.html
from constant_openai import OPENAI_API_KEY

# Set the environment
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

################ Simple Q&A Tasks ###############

# Initialize the model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

# Ask model a question
llm.invoke('How many Oscars the movie Oppenheimer received?')
### Note that: this is something that wasn't present in the training data of our model, so it shouldn't have a very accurate response

#llm.invoke('How many Oscars the movie Oppenheimer received? Who is the director of the movie?')

# We can visualize better the response by selecting only the content of the AIMessage
type(llm.invoke('How many Oscars the movie Oppenheimer received?')) # view that is a AIMessage

llm.invoke('How many Oscars the movie Oppenheimer received?').content

# Let's guide model's response with a prompt template (used to convert raw user input to a better input to the LLM)
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are member of the Academy of Motion Picture Arts and Sciences and a great film critic."),
    ("user", "{input}")
])

# We combine the prompt and the model into a simple LLM chain
chain = prompt | llm

# Now, we invoke it and ask our question again
chain.invoke({'input': "How many Oscars the movie Oppenheimer received?"}).content

### Note that: the output is a message, as our model is a ChatModel. One could add a output parser to convert the chat message to a string
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

# Just repeting the processes, but our output now is gonna be a string
chain = prompt | llm | output_parser

chain.invoke({'input': "How many Oscars the movie Oppenheimer received?"})



##################### Retrieval ######################

# In order to properly answer the original question, we need to provide additional context to the LLM.
# This is why we use Retrieval. A retriever can be backed in anything (PDFs, CSV files, HTML etc.).
# The way we apply it is using vectorstores.

# As we construct our Q&A model, let's ask it what's retrieval
llm.invoke("What is retrieval in Large Language Models? Why one would use it to get better answers by a LLM?").content

# Now, let's focus on providing the context to get a better response
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader

url = "https://en.wikipedia.org/wiki/List_of_accolades_received_by_Oppenheimer_(film)"
loader = WebBaseLoader(url)
docs = loader.load()

#print(docs)

# Now, we need to index it into a vectorstore. This is done by an embedding model first..
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# .. And using a vectorstore (in this case, FAISS) to store the data into vectors
from langchain_community.vectorstores import FAISS 
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
#print(documents)
vector = FAISS.from_documents(documents, embeddings)

# Since we stored our data, we can create a Retrieval Chain, which will take an incoming question, look up to relevant data and
# pass those documents along with the original question to the LLM and ask it to answer the original question
from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context:
    
    <context>
    {context}
    </context>

    Question: {input}                                       
""")

document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

from langchain.chains import create_retrieval_chain

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "How many Oscars the movie Oppenheimer received?"})
#print(response)
#type(response)
print(response["answer"]) # Get the "answer" key from the dictionary (see that type(response) is a dict)

response = retrieval_chain.invoke({"input": "Summarize the Academy Awards Nominations for Oppenheimer and the categories that the movie won an award."})
print(response["answer"])


################# Using COPOM Minutes #######################
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader

url = "https://www.bcb.gov.br/publicacoes/atascopom"
loader = WebBaseLoader(url)
docs = loader.load()

print(docs)

### Note that: We receive a message "Essa página depende do javascript para abrir [...]". Therefore, let's gonna use a PDFLoader
from langchain_community.document_loaders import PyPDFLoader
url = "https://www.bcb.gov.br/content/copom/atascopom/Copom261-not20240320261.pdf"
pdf = PyPDFLoader(url)
ata = pdf.load_and_split()

#print(ata)
#type(ata)       # Note that it's a list
#print(ata[1])

# We initialize the embeddings module
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# And use a vectorstore (in this case, FAISS) to store the data into vectors
from langchain_community.vectorstores import FAISS 
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(ata)
#print(documents)
vector = FAISS.from_documents(documents, embeddings)

# Since we stored our data, we can create a Retrieval Chain, which will take an incoming question, look up to relevant data and
# pass those documents along with the question to the LLM and ask it to answer
from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_template("""
    Responda a próxima pergunta baseando-se somente no contexto providenciado:
    
    <context>
    {context}
    </context>

    Question: {input}                                       
""")

document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

from langchain.chains import create_retrieval_chain

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "O que foi dito sobre a inflação de serviços?"})
print(response["answer"]) # Get the "answer" key from the dictionary (see that type(response) is a dict)

response = retrieval_chain.invoke({"input": "Houve sinalização futura do ritmo de cortes da Taxa Selic? E quanto à reunião de junho?"})
print(response["answer"])

response = retrieval_chain.invoke({"input": "Quanto ao mercado de trabalho, o que foi analisado e concluído pela autoridade monetária?"})
print(response["answer"])

response = retrieval_chain.invoke({"input": "Se tivesse de classificar essa ata em 'hawkish' ou 'dovish', como o faria e por quê?"})
print(response["answer"])