from flask import Flask, render_template, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
from functools import wraps
from config import Config
from utils import (
    predict_toxicity,
    overall_score,
    toxicity_level,
    analyze_sentiment,
    model_aligned_highlight,
    generate_explanation,
    validate_text_input,
    generate_rephrase_suggestions,
    compare_toxicity
)

# =========================
# FLASK SETUP
# =========================

app = Flask(__name__)
app.config.from_object(Config)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[Config.RATE_LIMIT]
)

# Logging setup
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================
# ERROR HANDLERS
# =========================

@app.errorhandler(404)
def not_found(error):
    if request.path.startswith('/api/'):
        return jsonify({"error": "Endpoint not found"}), 404
    return render_template("index.html", error="Page not found"), 404

@app.errorhandler(429)
def rate_limit_handler(e):
    if request.path.startswith('/api/'):
        return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
    return render_template("index.html", error="Too many requests. Please slow down."), 429

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}", exc_info=True)
    if request.path.startswith('/api/'):
        return jsonify({"error": "Internal server error"}), 500
    return render_template("index.html", error="Something went wrong. Please try again."), 500

# =========================
# ROUTES
# =========================

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        try:
            text = request.form.get("text", "").strip()
            
            # Validate input
            is_valid, error_msg = validate_text_input(text)
            if not is_valid:
                error = error_msg
                logger.warning(f"Invalid input: {error_msg}")
            else:
                logger.info(f"Analyzing text of length {len(text)}")
                
                # Perform analysis
                scores = predict_toxicity(text)
                overall = overall_score(scores)
                level = toxicity_level(overall)

                highlighted, contributing_words = model_aligned_highlight(text)
                sentiment = analyze_sentiment(text)
                explanation = generate_explanation(scores, len(contributing_words))

                # Generate rephrase suggestions for all negative comments
                suggestions = []
                if level in ["High", "Medium", "Low"]:
                    suggestions = generate_rephrase_suggestions(text, num_suggestions=3)
                    # Add comparison for each suggestion
                    for suggestion in suggestions:
                        comparison = compare_toxicity(text, suggestion["text"])
                        if comparison:
                            suggestion["comparison"] = comparison

                result = {
                    "text": text,  # Store original text
                    "highlighted": highlighted,
                    "scores": scores,
                    "overall": overall,
                    "level": level,
                    "sentiment": sentiment,
                    "explanation": explanation,
                    "suggestions": suggestions
                }
                
                logger.info(f"Analysis complete: level={level}, overall={overall}")
                
        except Exception as e:
            error = "Failed to analyze text. Please try again."
            logger.error(f"Analysis error: {e}", exc_info=True)

    return render_template("index.html", result=result, error=error)

@app.route("/api/analyze", methods=["POST"])
@limiter.limit("10 per minute")
def api_analyze():
    """API endpoint for toxicity analysis"""
    try:
        # Get JSON data
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = data["text"].strip()
        
        # Validate input
        is_valid, error_msg = validate_text_input(text)
        if not is_valid:
            return jsonify({"error": error_msg}), 400
        
        logger.info(f"API: Analyzing text of length {len(text)}")
        
        # Perform analysis
        scores = predict_toxicity(text)
        overall = overall_score(scores)
        level = toxicity_level(overall)
        
        # Optional detailed analysis
        include_details = data.get("include_details", False)
        
        response = {
            "scores": scores,
            "overall": overall,
            "level": level,
        }
        
        if include_details:
            highlighted, contributing_words = model_aligned_highlight(text)
            sentiment = analyze_sentiment(text)
            explanation = generate_explanation(scores, len(contributing_words))
            
            response.update({
                "sentiment": sentiment,
                "explanation": explanation,
                "contributing_words": contributing_words
            })
        
        logger.info(f"API: Analysis complete: level={level}, overall={overall}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"API error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route("/api/rephrase", methods=["POST"])
@limiter.limit("15 per minute")
def api_rephrase():
    """API endpoint for generating rephrase suggestions"""
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = data["text"].strip()
        num_suggestions = data.get("num_suggestions", 3)
        
        # Validate input
        is_valid, error_msg = validate_text_input(text)
        if not is_valid:
            return jsonify({"error": error_msg}), 400
        
        logger.info(f"API: Generating rephrase suggestions for text of length {len(text)}")
        
        # Generate suggestions
        suggestions = generate_rephrase_suggestions(text, num_suggestions)
        
        # Optionally compare each suggestion
        if data.get("include_comparison", True):
            for suggestion in suggestions:
                comparison = compare_toxicity(text, suggestion["text"])
                if comparison:
                    suggestion["comparison"] = comparison
        
        response = {
            "original_text": text,
            "suggestions": suggestions,
            "count": len(suggestions)
        }
        
        logger.info(f"API: Generated {len(suggestions)} suggestions")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"API rephrase error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/compare", methods=["POST"])
@limiter.limit("15 per minute")
def api_compare():
    """API endpoint for comparing toxicity between two texts"""
    try:
        data = request.get_json()
        if not data or "original" not in data or "rephrased" not in data:
            return jsonify({"error": "Missing 'original' or 'rephrased' field"}), 400
        
        original = data["original"].strip()
        rephrased = data["rephrased"].strip()
        
        # Validate inputs
        is_valid, error_msg = validate_text_input(original)
        if not is_valid:
            return jsonify({"error": f"Original text: {error_msg}"}), 400
        
        is_valid, error_msg = validate_text_input(rephrased)
        if not is_valid:
            return jsonify({"error": f"Rephrased text: {error_msg}"}), 400
        
        logger.info(f"API: Comparing texts")
        
        comparison = compare_toxicity(original, rephrased)
        
        if not comparison:
            return jsonify({"error": "Comparison failed"}), 500
        
        logger.info(f"API: Comparison complete, improvement: {comparison.get('improvement_percent')}%")
        return jsonify(comparison), 200
        
    except Exception as e:
        logger.error(f"API compare error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    logger.info(f"Starting app on {Config.HOST}:{Config.PORT}, debug={Config.DEBUG}")
    app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)

