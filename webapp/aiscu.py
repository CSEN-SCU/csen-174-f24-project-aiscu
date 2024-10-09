# Virtual env settings:
# conda create -n webapp ipython
# conda activate webapp
# pip install langchain pinecone-client openai tiktoken nest_asyncio langchain-pinecone pinecone-notebooks langchain-openai langchain_community beautifulsoup4

import os
os.environ["OPENAI_API_KEY"] = "" #OPENAI API KEY#

from pinecone import Pinecone
pc = Pinecone(api_key="") #PINECONE API KEY#

# fixes a bug with asyncio and jupyter
import nest_asyncio
nest_asyncio.apply()

from langchain_community.document_loaders import WebBaseLoader
# List of URLs you want to load from
urls = [
    "https://www.scu.edu/drahmann/tutoring/",
    "https://www.scu.edu/cas/mathematics-learning-center/",
    "https://www.scu.edu/provost/writingcenter/"
    "https://www.scu.edu/engineering/undergraduate/student-support/tau-beta-pi-tutoring/"
    # Add more URLs as needed
]
# Create a WebBaseLoader to scrape the provided URLs
loader = WebBaseLoader(urls)
loader.requests_per_second = 1
docs = loader.aload()
# Optionally, you can now process the docs as needed
#print(docs)

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1200,
    chunk_overlap  = 200,
    length_function = len,
)
docs_chunks = text_splitter.split_documents(docs)
#print(docs_chunks)

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
#print(embeddings)

from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
import time
index_name = "langchain-test-index"  # change if desired
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
index = pc.Index(index_name)
docsearch = PineconeVectorStore(index=index, embedding=embeddings)

from uuid import uuid4
uuids = [str(uuid4()) for _ in range(len(docs_chunks))]
docsearch.add_documents(documents=docs_chunks, ids=uuids)
#print(docsearch)

#BELOW 2 LINES NOT USED#
# from langchain.vectorstores import Chroma
# docsearch = Chroma.from_documents(docs, embeddings)

#results = docsearch.similarity_search(
#    "I'm struggling with my Phys 32 class. Any suggestions?",
#    k=1  # Remove the filter parameter
#)
#for res in results:
#    print(f"* {res.page_content} [{res.metadata}]")

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
llm=OpenAI()
qa_with_sources = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)

import sys
query = sys.argv[1] #"I have a Math 11 midterm coming up. Is there any place I can get help?"#

result = qa_with_sources({"query": query})
print(result["result"])
#print(result["source_documents"])