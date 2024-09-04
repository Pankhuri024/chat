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
        # Initialize OpenAI model
        selected_model = 'gpt-3.5-turbo'  # Example model, adjust as needed
        llm = ChatOpenAI(model=selected_model)
        
        # Define your template
        template = """Analyze the content of the provided text and generate insights. Include a summary (200 characters) and a detailed description (1500 characters). Output the response in JSON format without 'json' heading, with each insight structured as follows and {input}:

        - Insight:
          - Summary: Insight summary here
          - Description: Detailed insight description here

        Instructions:
        1. Base your response solely on the content within the provided text.
        2. Do not introduce new elements or information not present in the text.
        3. If there is no insight, generate the response without JSON header with the message: "Message": "There is no insight found. Please send a different text."
        4. Ensure the response does not mention ChatGPT or OpenAI.
        """
        
        # Construct prompt using the template and the question
        prompt = template.format(input=question)
        
        # Send the prompt to the model
        response = llm(prompt)
        
        # Extract 'Insights' from the response
        insights = response.get('content', '{}')
        try:
            insights_json = json.loads(insights)  # Convert string to dictionary
        except json.JSONDecodeError as e:
            logging.error(f"JSON Decode Error: {e}")
            return jsonify({'message': 'Error decoding JSON response'}), 500
        
        insights_result = insights_json.get('Insights', {})
        
        return jsonify(insights_result)
    
    except Exception as e:
        if "insufficient_quota" in str(e):
            logging.error("Quota exceeded: Please check your OpenAI plan and billing details.")
            return jsonify({'message': 'Quota exceeded. Please check your OpenAI plan and billing details.'}), 429
        logging.error(f"Error processing text: {e}")
        return jsonify({'message': 'Error processing text'}), 500

if __name__ == '__main__':
    app.run(debug=True)
