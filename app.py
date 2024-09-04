import os
from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import json
import re
import logging

logging.basicConfig(level=logging.DEBUG,  # Change to DEBUG to get detailed logs
                    format='%(asctime)s - %(levelname)s - %(message)s')

models = "gpt-4o"

load_dotenv()

ALLOWED_IP_ADDRESSES = {"127.0.0.1", "171.50.226.59"}  # Add allowed IP addresses here

app = Flask(__name__)

def extract_number(key):
    match = re.search(r'\d+', key)
    return int(match.group()) if match else 0

@app.route('/ask-question', methods=['POST'])
def ask_question():
    logging.debug("Starting ask question function.")
    
    # Ensure OPENAI_API_KEY is set
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("The OPENAI_API_KEY environment variable is not set. Please set it in your environment.")
    logging.debug(f"OPENAI_API_KEY: {OPENAI_API_KEY}")

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    question = request.args.get('question', '')
    if not question:
        return jsonify({'Message': 'No question provided'}), 400

    try:
        # Initialize components for a fresh session
        selected_model = models
        llm = ChatOpenAI(model=selected_model)

        template_with_question_for_insights = """Analyze the content of the provided text and generate insights. Include a summary (200 characters) and a detailed description (1500 characters). Output the response in JSON format structured as follows without JSON headers:

- Insight:
- Summary: Insight summary here
- Description: Detailed insight description here

Instructions:
1. Base your response solely on the content within the provided text.
2. Do not introduce new elements or information not present in the text.
3. If there is no insight, generate the response without JSON headers with the message: "Message": "There is no insight found. Please send a different text."
4. Ensure the response does not mention ChatGPT or OpenAI.
"""

        # Construct prompt template
        logging.debug('Constructing prompt template.')
        custom_rag_prompt = PromptTemplate.from_template(template_with_question_for_insights)
        
        document_chain = create_stuff_documents_chain(llm, custom_rag_prompt)
        response = document_chain.invoke({"input": question})

        try:
            # Load the data from the JSON response
            data = json.loads(response["answer"])

            # Order the dictionary by its keys
            ordered_data = dict(sorted(data.items(), key=lambda item: extract_number(item[0])))

            # Convert the ordered dictionary back to JSON format
            ordered_data = json.dumps(ordered_data, indent=4)
        except json.JSONDecodeError:
            return jsonify({"Message": "Failed to decode JSON response from LLM"}), 500

        if not data:
            return jsonify({"Message": "There is no valid response found."}), 200

        # Clear variables to avoid retention of previous data
        del llm
        del document_chain
        del response
        del selected_model

        # Return response
        return Response(ordered_data, content_type='application/json'), 200
    except Exception as e:
        return jsonify({"Message": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0")
