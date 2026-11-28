from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

vectorstore=Chroma(
    persist_directory="./vectorstore",
    embedding_function=embeddings

)

query=input("Enter a question to search the PDF: ")

results=vectorstore.similarity_search_with_score(
    query,
    k=3
)


#display results

for i, (documents,distance) in enumerate(results):
    print(f"Result {i+1}:")
    print("Page content:", documents.page_content)
    page=documents.metadata.get("page")

    if isinstance(page, int):
        print("page:", page+1)

    print("Distance:", distance)
    print("\n")