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
    
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'message': 'Invalid text'}), 400

    try:
        # Initialize OpenAI model
        selected_model = 'gpt-3.5-turbo'  # Example model, adjust as needed
        llm = ChatOpenAI(model=selected_model)
        
        # Define your template
        template = """
Analyze the content of the provided text and generate insights. Include a summary (200 characters) and a detailed description (1500 characters). Output the response in JSON format without a 'json' heading, with each insight structured as follows and based on the provided input:

- Insight:
  - Summary: Insight summary here
  - Description: Detailed insight description here

Instructions:
1. Base your response solely on the content within the provided text.
2. Do not introduce new elements or information not present in the text.
3. If there is no insight, generate the response without JSON header with the message: "Message": "There is no insight found. Please send a different text."
4. Ensure the response does not mention ChatGPT or OpenAI.
{input}
        """
        
        # Construct prompt using the template and the text
        prompt = template.format(input=text)
        
        # Send the prompt to the model
        response = llm(prompt)
        
        # Extract the AI response
        response_text = response['choices'][0]['message']['content'] if 'choices' in response else str(response)

        # Return the response text directly
        return Response(response_text, mimetype='application/json')
    
    except Exception as e:
        if "insufficient_quota" in str(e):
            logging.error("Quota exceeded: Please check your OpenAI plan and billing details.")
            return jsonify({'message': 'Quota exceeded. Please check your OpenAI plan and billing details.'}), 429
        logging.error(f"Error processing text: {e}")
        return jsonify({'message': 'Error processing text'}), 500

if __name__ == '__main__':
    app.run(debug=True)
