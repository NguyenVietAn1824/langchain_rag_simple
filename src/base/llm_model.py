import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def get_llm(Vector_database):
    load_dotenv()
    api_key = os.environ.get('OPENAI_API_KEY')
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    
    Vector_database = Vector_database()
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer."""

    custom_rag_prompt = PromptTemplate.from_template(template)


    rag_chain = (
    {"context": Vector_database. | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
    )

