import torch
import re
import logging
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import Config

# =========================
# LOGGING SETUP
# =========================

logger = logging.getLogger(__name__)

# =========================
# MODEL SETUP (CACHED)
# =========================

_model = None
_tokenizer = None

def get_model():
    """Lazy load and cache model"""
    global _model
    if _model is None:
        try:
            logger.info(f"Loading model from {Config.MODEL_PATH}")
            _model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_PATH)
            _model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    return _model

def get_tokenizer():
    """Lazy load and cache tokenizer"""
    global _tokenizer
    if _tokenizer is None:
        try:
            logger.info(f"Loading tokenizer from {Config.MODEL_PATH}")
            _tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise RuntimeError(f"Tokenizer loading failed: {e}")
    return _tokenizer

# =========================
# INPUT VALIDATION
# =========================

def validate_text_input(text):
    """Validate text input for analysis"""
    if not text or not text.strip():
        return False, "Text cannot be empty"
    
    if len(text) < Config.MIN_TEXT_LENGTH:
        return False, f"Text must be at least {Config.MIN_TEXT_LENGTH} character(s)"
    
    if len(text) > Config.MAX_TEXT_LENGTH:
        return False, f"Text must not exceed {Config.MAX_TEXT_LENGTH} characters"
    
    # Check for valid characters (prevent binary/malicious input)
    if not text.isprintable() and not any(c in text for c in ['\n', '\r', '\t']):
        return False, "Text contains invalid characters"
    
    return True, None

# =========================
# TOXICITY PREDICTION
# =========================

def predict_toxicity(text):
    """Predict toxicity scores for given text"""
    try:
        tokenizer = get_tokenizer()
        model = get_model()
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=Config.MAX_TOKENS
        )

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.sigmoid(logits)[0].tolist()
        return dict(zip(Config.LABELS, [round(p, 2) for p in probs]))
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise

def overall_score(scores):
    """Calculate overall toxicity score"""
    return round(max(scores.values()), 2)

def toxicity_level(score):
    """Determine toxicity level from score"""
    if score < Config.TOXICITY_LOW_THRESHOLD:
        return "Low"
    elif score < Config.TOXICITY_MEDIUM_THRESHOLD:
        return "Medium"
    else:
        return "High"

# =========================
# SENTIMENT (OFFLINE)
# =========================

def analyze_sentiment(text):
    """Analyze sentiment using keyword matching"""
    try:
        words = re.findall(r"[a-zA-Z]+", text.lower())
        pos = sum(1 for w in words if w in Config.POSITIVE_WORDS)
        neg = sum(1 for w in words if w in Config.NEGATIVE_WORDS)

        if pos > neg:
            return "Positive"
        elif neg > pos:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return "Unknown"

# =========================
# MODEL-ALIGNED WORD HIGHLIGHTING (OPTIMIZED)
# =========================

def model_aligned_highlight(text, drop_threshold=None):
    """Highlight words that contribute to toxicity (optimized)"""
    if drop_threshold is None:
        drop_threshold = Config.TOXICITY_DROP_THRESHOLD
    
    try:
        words = text.split()
        
        # Limit processing for very long texts
        if len(words) > 100:
            logger.warning(f"Text has {len(words)} words, using first 100 for highlighting")
            words = words[:100]
            text = " ".join(words)

        base_scores = predict_toxicity(text)
        base_overall = overall_score(base_scores)

        highlighted = []
        contributing_words = []

        # Batch predictions for better performance
        for i, word in enumerate(words):
            masked = words[:i] + words[i+1:]
            masked_text = " ".join(masked)

            masked_scores = predict_toxicity(masked_text)
            masked_overall = overall_score(masked_scores)

            if base_overall - masked_overall >= drop_threshold:
                highlighted.append(f"<span class='toxic'>{word}</span>")
                contributing_words.append(word)
            else:
                highlighted.append(word)

        return " ".join(highlighted), contributing_words
    
    except Exception as e:
        logger.error(f"Highlighting error: {e}", exc_info=True)
        # Return unhighlighted text on error
        return text, []

# =========================
# EXPLANATION
# =========================

def generate_explanation(scores, contributing_count):
    """Generate human-readable explanation"""
    try:
        reasons = [
            label.replace("_", " ")
            for label, score in scores.items()
            if score >= Config.TOXICITY_HIGH_THRESHOLD
        ]

        if not reasons and contributing_count == 0:
            return "No significant toxic patterns were detected by the model."

        explanation = "The model flagged this comment due to "

        if reasons:
            explanation += "high levels of " + ", ".join(reasons)

        if contributing_count > 0:
            if reasons:
                explanation += " and "
            explanation += f"{contributing_count} word(s) that strongly influenced the prediction"

        return explanation.capitalize() + "."
    
    except Exception as e:
        logger.error(f"Explanation generation error: {e}")
        return "Unable to generate explanation."

# =========================
# REPHRASE SUGGESTIONS
# =========================

# Toxic word mappings for replacement
TOXIC_REPLACEMENTS = {
    # Profanity
    "hate": "dislike", "damn": "darn", "stupid": "unwise", "idiot": "inexperienced",
    "dumb": "mistaken", "crap": "nonsense", "sucks": "disappointing", "awful": "subpar",
    "terrible": "poor", "worst": "weakest", "horrible": "unfortunate", "disgusting": "unpleasant",
    "pathetic": "inadequate", "loser": "person", "moron": "individual", "jerk": "person",
    "trash": "poor quality", "garbage": "low quality", "useless": "ineffective",
    
    # Aggressive terms
    "kill": "stop", "destroy": "change", "annihilate": "remove", "attack": "critique",
    "fight": "discuss", "war": "debate", "enemy": "opponent", "battle": "compete",
    
    # Insulting adjectives
    "ugly": "unattractive", "fat": "overweight", "lazy": "inactive", "crazy": "unusual",
    "insane": "extreme", "ridiculous": "questionable", "absurd": "unlikely",
}

# Toxic patterns and their neutral equivalents
PATTERN_REPLACEMENTS = [
    (r'\byou are (stupid|dumb|an idiot)\b', r'that perspective seems mistaken'),
    (r'\bshut up\b', r'please stop'),
    (r'\bgo to hell\b', r'please leave'),
    (r'\bwhat the (hell|heck)\b', r'what is happening'),
    (r'\bi hate you\b', r'I disagree with you'),
    (r'\byou suck\b', r'I disagree'),
    (r'\bget lost\b', r'please go away'),
]

def generate_rephrase_suggestions(text, num_suggestions=3):
    """Generate multiple rephrase suggestions to reduce toxicity"""
    try:
        suggestions = []
        
        # Strategy 1: Replace toxic words with neutral alternatives
        suggestion1 = text.lower()
        replacements_made = []
        for toxic, neutral in TOXIC_REPLACEMENTS.items():
            if toxic in suggestion1:
                suggestion1 = suggestion1.replace(toxic, neutral)
                replacements_made.append(f"'{toxic}' → '{neutral}'")
        
        if replacements_made:
            suggestions.append({
                "text": suggestion1.capitalize(),
                "strategy": "Word Replacement",
                "changes": replacements_made[:3],  # Show first 3 changes
                "description": "Replaced offensive words with neutral alternatives"
            })
        
        # Strategy 2: Pattern-based rephrasing
        suggestion2 = text
        pattern_changes = []
        for pattern, replacement in PATTERN_REPLACEMENTS:
            if re.search(pattern, suggestion2, re.IGNORECASE):
                suggestion2 = re.sub(pattern, replacement, suggestion2, flags=re.IGNORECASE)
                pattern_changes.append("Rephrased offensive expression")
        
        if pattern_changes and suggestion2 != text:
            suggestions.append({
                "text": suggestion2,
                "strategy": "Expression Rewrite",
                "changes": pattern_changes[:2],
                "description": "Restructured aggressive expressions"
            })
        
        # Strategy 3: Tone softening (add polite qualifiers)
        suggestion3 = text
        if any(word in text.lower() for word in ["you", "your", "this", "that"]):
            # Add softening phrases
            if suggestion3[-1] in ['.', '!']:
                suggestion3 = suggestion3[:-1]
            
            softeners = [
                " I think.", " in my opinion.", " perhaps.", " possibly."
            ]
            suggestion3 = suggestion3 + softeners[len(suggestions) % len(softeners)]
            
            # Remove excessive punctuation
            suggestion3 = re.sub(r'!+', '.', suggestion3)
            suggestion3 = re.sub(r'\?+', '?', suggestion3)
            
            suggestions.append({
                "text": suggestion3,
                "strategy": "Tone Softening",
                "changes": ["Added qualifying language", "Reduced emphatic punctuation"],
                "description": "Made language more tentative and polite"
            })
        
        # Strategy 4: Question transformation (convert statements to questions)
        if not text.strip().endswith('?'):
            suggestion4 = text.strip()
            if suggestion4[-1] in ['.', '!']:
                suggestion4 = suggestion4[:-1]
            
            # Replace some aggressive words
            for toxic, neutral in list(TOXIC_REPLACEMENTS.items())[:5]:
                suggestion4 = suggestion4.replace(toxic, neutral)
            
            suggestion4 = f"Could we discuss {suggestion4.lower()}?"
            
            suggestions.append({
                "text": suggestion4,
                "strategy": "Question Format",
                "changes": ["Converted to question", "Invited dialogue"],
                "description": "Transformed statement into constructive question"
            })
        
        # If no suggestions generated, create a generic one
        if not suggestions:
            generic = text
            # Remove excessive punctuation
            generic = re.sub(r'!+', '.', generic)
            generic = re.sub(r'\.{2,}', '.', generic)
            
            suggestions.append({
                "text": generic,
                "strategy": "Punctuation Cleanup",
                "changes": ["Normalized punctuation"],
                "description": "Reduced emphatic punctuation"
            })
        
        return suggestions[:num_suggestions]
    
    except Exception as e:
        logger.error(f"Rephrase generation error: {e}", exc_info=True)
        return []

def compare_toxicity(original_text, rephrased_text):
    """Compare toxicity between original and rephrased text"""
    try:
        original_scores = predict_toxicity(original_text)
        original_overall = overall_score(original_scores)
        
        rephrased_scores = predict_toxicity(rephrased_text)
        rephrased_overall = overall_score(rephrased_scores)
        
        improvement = original_overall - rephrased_overall
        improvement_percent = (improvement / original_overall * 100) if original_overall > 0 else 0
        
        return {
            "original": {
                "scores": original_scores,
                "overall": original_overall,
                "level": toxicity_level(original_overall)
            },
            "rephrased": {
                "scores": rephrased_scores,
                "overall": rephrased_overall,
                "level": toxicity_level(rephrased_overall)
            },
            "improvement": round(improvement, 2),
            "improvement_percent": round(improvement_percent, 1),
            "is_better": improvement > 0
        }
    
    except Exception as e:
        logger.error(f"Comparison error: {e}", exc_info=True)
        return None

