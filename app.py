import os
import logging
from flask import Flask, request, jsonify, abort, Response
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.DEBUG,  # Change to DEBUG to get detailed logs
                    format='%(asctime)s - %(levelname)s - %(message)s')

models = "gpt-4o"

load_dotenv()

app = Flask(__name__)

@app.route('/ask-question', methods=['POST'])
def ask_question():
    logging.debug("Starting ask_question function.")
    
    # Ensure OPENAI_API_KEY is set
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("The OPENAI_API_KEY environment variable is not set. Please set it in your environment.")
    logging.debug(f"OPENAI_API_KEY: {OPENAI_API_KEY}")

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({"Message": "No question provided"}), 400

    try:
        # Initialize LLM
        selected_model = models
        llm = ChatOpenAI(model=selected_model)

        template_for_insights = """
Analyze the content of the provided text and generate insights. Include a summary (200 characters) and a detailed description (1500 characters). Output the response in JSON format structured as follows:

- Insight:
- Summary: Insight summary here
- Description: Detailed insight description here

Instructions:
1. Base your response solely on the content within the provided text.
2. Do not introduce new elements or information not present in the text.
3. If there is no insight, generate the response with the message: "Message": "There is no insight found. Please send a different text."
4. Ensure the response does not mention ChatGPT or OpenAI.
{input}
"""


        # Construct prompt template
        logging.debug('Constructing prompt template.')
        custom_rag_prompt = PromptTemplate.from_template(template_for_insights)
        document_chain = create_stuff_documents_chain(llm, custom_rag_prompt)
        
        response = document_chain.invoke({"input": question})
        
        return jsonify({"Answer": response['answer']}), 200
    except Exception as e:
        logging.error(f"Error processing question: {e}")
        return jsonify({"Message": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0")
