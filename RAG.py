import os
import langchain_openai
import langchain
import openai
import requests
from langchain_community.document_loaders.text import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
#from langchain_community.vectorstores import Weaviate
#import weaviate
#from weaviate.embedded import EmbeddedOptions
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables.passthrough import RunnablePassthrough
#from constant import OPENAI_API_KEY

# Setting the Open AI API key
#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = "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"

# Get Request of the document (text file)
url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/modules/state_of_the_union.txt"
req = requests.get(url)
with open("state_of_the_union.txt", "w") as f:
    f.write(req.text)

print(f)
type(f)

loader = TextLoader('./state_of_the_union.txt')
documents = loader.load()
type(documents)
print(documents)

# Chunking the documents
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(chunks)
type(chunks)

# Embedding and storing the chunks using Open AI Embedding model and Weviate
#client = weaviate.Client(
#    embedded_options=EmbeddedOptions
#)

#vectorstore = Weaviate.from_documents(
#    client=client,
#    documents=chunks,
#    embedding=OpenAIEmbeddings(),
#   by_text=False
#)

# Embedding using Open AI Embedding Model
vector_db = FAISS.from_documents(chunks, OpenAIEmbeddings())

# Retrieving
#retriever = vectorstore.as_retriever()
retriever = vector_db.as_retriever()

# Augmenting
template = """You're an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say you don't know.
Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
print(prompt)

# Generating
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#query = "What did the president say about Justice Breyer?"
query = "What did the president say about Ukraine?"
rag_chain.invoke(query)



################################### Com Raspagem de Dados #################################
import bs4
import requests
from langchain_community.document_loaders.text import TextLoader
import pandas as pd

url = "https://www.whitehouse.gov/briefing-room/speeches-remarks/2024/03/07/remarks-of-president-joe-biden-state-of-the-union-address-as-prepared-for-delivery-2/"

response = requests.get(url)
soup = bs4.BeautifulSoup(response.content, "html.parser")
print(response)
print(soup)

##transcript_element = page.find('section', class_='body-content')
##print(transcript_element)
##type(transcript_element)

##text_content = transcript_element.get_text(separator='\n').strip()
##print(text_content)

transcript_element = soup.find_all('p')[2:]
print(transcript_element)

#transcript_data = [data.text.strip().split('\xa0') for data in transcript_element]
transcript_data = [data.text.strip() for data in transcript_element]
print(transcript_data)

df = pd.DataFrame(transcript_data, columns=['Text'])
df
df.replace(to_replace=[''], value=[''], regex=True, inplace=True)
df.to_csv('biden.csv')



from langchain_community.document_loaders import BSHTMLLoader
loader = BSHTMLLoader("")


