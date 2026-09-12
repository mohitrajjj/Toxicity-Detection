import os

# =========================
# FLASK CONFIGURATION
# =========================

class Config:
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    PORT = int(os.environ.get('PORT', 5001))
    HOST = os.environ.get('HOST', '127.0.0.1')
    
    # Model paths
    MODEL_PATH = os.environ.get('MODEL_PATH', 'bert_toxic_model')
    
    # Analysis settings
    MAX_TEXT_LENGTH = 500
    MIN_TEXT_LENGTH = 1
    MAX_TOKENS = 128
    
    # Toxicity thresholds
    TOXICITY_DROP_THRESHOLD = 0.04  # For word highlighting
    TOXICITY_LOW_THRESHOLD = 0.3
    TOXICITY_MEDIUM_THRESHOLD = 0.7
    TOXICITY_HIGH_THRESHOLD = 0.6  # For explanations
    
    # Rate limiting
    RATE_LIMIT = "20 per minute"
    
    # Labels
    LABELS = [
        "toxicity",
        "severe_toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack"
    ]
    
    # Sentiment analysis
    POSITIVE_WORDS = {
        "good", "great", "love", "nice", "excellent", "amazing",
        "wonderful", "fantastic", "awesome", "brilliant", "perfect"
    }
    NEGATIVE_WORDS = {
        "bad", "hate", "awful", "terrible", "worst", "angry",
        "horrible", "disgusting", "pathetic", "stupid", "idiot"
    }
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = 'app.log'
