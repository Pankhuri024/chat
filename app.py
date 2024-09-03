from flask import Flask, jsonify, request
import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/api/send-text', methods=['POST'])
def receive_text():
    logging.debug("Starting receive_text function.")
    
    # Ensure OPENAI_API_KEY is set
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY environment variable is not set.")
        return jsonify({'message': 'OPENAI_API_KEY environment variable is not set.'}), 500
    
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    logging.debug(f"OPENAI_API_KEY: {OPENAI_API_KEY}")

    data = request.get_json()
    question = data.get('text', '')
    
    if not question:
        return jsonify({'message': 'Invalid text'}), 400

    try:
        # Initialize the language model
        selected_model = 'gpt-3.5-turbo'  # Model to use
        llm = ChatOpenAI(model=selected_model)

        # Define a prompt template
        template = """
        You will receive a question and you should respond to it based on the provided context:

        Question: {input}

        Please provide a detailed answer that covers all aspects of the question.
        """
        custom_prompt = PromptTemplate.from_template(template)

        # Use the model to generate a response
        prompt_text = custom_prompt.render({"input": question})
        logging.debug(f"Generated prompt for model: {prompt_text}")

        response = llm.invoke({"prompt": prompt_text})

        return jsonify({'question': question, 'response': response, 'message': 'Text processed successfully'})
    
    except Exception as e:
        logging.error(f"Error processing text: {e}")
        return jsonify({'message': 'Error processing text'}), 500

if __name__ == '__main__':
    app.run(debug=True)
