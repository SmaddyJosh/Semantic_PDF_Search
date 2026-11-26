from langchain_community.document_loaders import PyPDFLoader

pdf_path="data/Reg.pdf"

loader = PyPDFLoader(pdf_path)

#extract pdf content
documents=loader.load()

print("Number of pages in the PDF:", len(documents))

for document in documents:
    print("Page content:", document.page_content[:500])

    print("Metadata:", document.metadata)