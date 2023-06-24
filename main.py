!pip install transformers

from transformers import BartTokenizer, BartForConditionalGeneration

def summarize_text(text):
    # Load the BART tokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

    # Load the BART model
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    # Tokenize the input text
    inputs = tokenizer([text], max_length=1024, truncation=True, return_tensors='pt')

    # Generate the summary
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

    return summary

# Get user input
user_text = input("Please enter the text that you would like to SummaRise: ")

# Summarize the user's text
summary = summarize_text(user_text)

# Print the summary
print("Summary:")
print(summary)
