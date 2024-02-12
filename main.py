import logging
import torch
from flask import Flask, request, jsonify
from transformers import BartTokenizer, BartForConditionalGeneration

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load the tokenizer and model only once, at startup, to save resources.
try:
    logger.info("Loading BART model and tokenizer...")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    model.eval()  # Set model to evaluation mode for inference
    if torch.cuda.is_available():
        model.to('cuda')
        logger.info("Using CUDA for model inference.")
    else:
        logger.info("CUDA not available. Using CPU for model inference.")
except Exception as e:
    logger.error(f"Failed to load model/tokenizer: {e}")
    raise e

def summarize_text(text, max_input_length=1024, max_summary_length=150, num_beams=4):
    """
    Summarizes the provided text using the BART model.
    
    Parameters:
        text (str): The text to summarize.
        max_input_length (int): Max length for the input text.
        max_summary_length (int): Max length of the generated summary.
        num_beams (int): Number of beams for beam search.
    
    Returns:
        str: The summarized text.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = tokenizer([text], max_length=max_input_length, return_tensors='pt', truncation=True).to(device)
    with torch.no_grad():
        summary_ids = model.generate(inputs['input_ids'], num_beams=num_beams, max_length=max_summary_length, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@app.route('/summarize', methods=['POST'])
def summarize_route():
    """
    API endpoint for text summarization.
    
    Expects a JSON payload with a 'text' field.
    """
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided for summarization.'}), 400
    
    try:
        summary = summarize_text(text)
        return jsonify({'summary': summary}), 200
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        return jsonify({'error': 'Failed to summarize the text.'}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
