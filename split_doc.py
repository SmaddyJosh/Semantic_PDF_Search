#Divide doc into smaller pieces(chunks)
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

loader=PyPDFLoader("data/Reg.pdf")
documents=loader.load()

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=100
)

chunks=text_splitter.split_documents(documents)

print("Number of pages in the PDF:", len(documents))

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} content:", chunk.page_content[:500])
    print(f"Chunk {i+1} metadata:", chunk.metadata)