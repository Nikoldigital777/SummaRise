import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from transformers import pipeline, Pipeline
import torch

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Text Summarization API")

# Define the request body with detailed validation
class SummarizationRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Text to summarize; must be at least 10 characters long.")
    max_length: int = Field(150, ge=50, le=300, description="Maximum length of the summary.")
    min_length: int = Field(40, ge=20, le=150, description="Minimum length of the summary.")
    num_beams: int = Field(4, ge=1, le=10, description="Number of beams for beam search.")

# Load the summarization pipeline with error handling
def load_summarizer() -> Pipeline:
    try:
        logger.info("Loading summarization pipeline...")
        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"{'Using CUDA' if device == 0 else 'CUDA not available. Using CPU'} for model inference.")
        
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
        logger.info("Summarization pipeline loaded successfully.")
        return summarizer
    except Exception as e:
        logger.critical("Failed to load the summarization pipeline", exc_info=True)
        raise RuntimeError("Failed to load summarization model. Please check model availability or dependencies.") from e

summarizer = load_summarizer()

@app.post("/summarize", response_model=dict)
async def summarize(request: SummarizationRequest):
    """
    API endpoint for text summarization.

    Expects a JSON payload with a 'text' field and optional parameters.
    """
    text = request.text.strip()
    if not text:
        logger.warning("Empty text received for summarization")
        raise HTTPException(status_code=400, detail="No text provided for summarization.")
    
    max_input_length = 1024
    if len(text) > max_input_length:
        logger.warning("Text too long for summarization endpoint")
        raise HTTPException(status_code=400, detail=f"Input text is too long. Maximum length is {max_input_length} characters.")

    try:
        summary_list = summarizer(
            text,
            max_length=request.max_length,
            min_length=request.min_length,
            num_beams=request.num_beams,
            truncation=True,
        )
        summary = summary_list[0]['summary_text']
        logger.info("Summarization completed successfully.")
        return {"summary": summary}
    except ValidationError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=422, detail="Invalid request parameters.")
    except Exception as e:
        logger.error(f"Unexpected error during summarization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to summarize the text.")

# Run the app using Uvicorn with environment-based configuration
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    log_level = os.getenv("LOG_LEVEL", "info")
    uvicorn.run("your_script_name:app", host=host, port=port, log_level=log_level)
