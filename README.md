# Toxicity Detection App

AI-powered toxic comment detection and analysis using BERT.

## Features

✅ **Real-time Toxicity Analysis** - Detect toxic, obscene, threatening, and insulting content  
✅ **Model-Aligned Word Highlighting** - See which words contribute to toxicity  
✅ **Sentiment Analysis** - Understand the emotional tone  
✅ **AI-Powered Rephrase Suggestions** - Get alternative phrasings to reduce toxicity  
✅ **Before/After Comparison** - See toxicity improvement in real-time  
✅ **Multiple Rephrasing Strategies** - Word replacement, tone softening, expression rewrite  
✅ **📊 Batch Analysis** - Analyze multiple comments at once (up to 20)  
✅ **📈 Analytics Dashboard** - Visual statistics and toxicity distribution  
✅ **📥 Export Results** - Download analysis results as CSV or JSON  
✅ **Interactive Charts** - Level distribution and category breakdown visualizations  
✅ **Interactive UI** - Loading indicators, character counter, copy/clear buttons  
✅ **REST API** - Programmatic access with rate limiting  
✅ **Robust Error Handling** - Graceful failures and validation  
✅ **Performance Optimized** - Model caching and efficient processing  

## Installation

1. **Clone or navigate to the project:**
   ```bash
   cd toxic_app
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model exists:**
   - The app requires a trained model in `bert_toxic_model/`
   - If not available, train using: `python train_model.py`

## Usage

### Running the Web App

```bash
python app.py
```

Visit: `http://127.0.0.1:5001`

### Configuration

Create a `.env` file or set environment variables:

```bash
FLASK_DEBUG=False              # Enable debug mode (default: False)
PORT=5001                      # Server port (default: 5001)
HOST=127.0.0.1                 # Server host (default: 127.0.0.1)
SECRET_KEY=your-secret-key     # Flask secret key
LOG_LEVEL=INFO                 # Logging level (default: INFO)
```

### Batch Analysis

The app now supports analyzing multiple comments at once:

1. Click **"📊 Batch Analysis"** tab
2. **Option A - Manual Entry**: Enter one comment per line (max 20 comments)
3. **Option B - File Import**: Click "📤 Choose File to Import" and upload:
   - **PDF files** (.pdf) - Extracts all text content
   - **Word documents** (.doc, .docx) - Extracts text from Word files
   - **CSV files** (.csv) - Reads comments from CSV (one per row or cell)
   - **Text files** (.txt) - Plain text, one comment per line
4. Click **"Analyze Batch"**
5. View comprehensive analytics:
   - Overall statistics (total, average toxicity, distribution)
   - Visual charts (level distribution, category breakdown)
   - Detailed results table
6. Export results as CSV or JSON

**File Import Notes:**
- Files are processed client-side (secure, no uploads to server)
- Text is automatically extracted and formatted (one comment per line)
- Large files may take a moment to process
- Supports multiple file formats for maximum flexibility

**Analytics Provided:**
- Total comments analyzed
- Average toxicity score
- Count by toxicity level (High/Medium/Low)
- Category-wise breakdown
- Individual comment details with toxicity scores

**Export Options:**
- **CSV**: Spreadsheet-compatible format with all metrics
- **JSON**: Structured data for further processing

### API Usage

**Endpoint:** `POST /api/analyze`

**Request:**
```json
{
  "text": "Your comment here",
  "include_details": true
}
```

**Response:**
```json
{
  "scores": {
    "toxicity": 0.85,
    "severe_toxicity": 0.12,
    "obscene": 0.34,
    "threat": 0.05,
    "insult": 0.67,
    "identity_attack": 0.03
  },
  "overall": 0.85,
  "level": "High",
  "sentiment": "Negative",
  "explanation": "The model flagged this comment due to high levels of toxicity, insult and 2 word(s) that strongly influenced the prediction.",
  "contributing_words": ["word1", "word2"]
}
```

**Rate Limits:**
- Web interface: 20 requests/minute
- API endpoint: 10 requests/minute

**Example cURL:**
```bash
curl -X POST http://127.0.0.1:5001/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test comment", "include_details": true}'
```

### Rephrase Suggestions API

**Endpoint:** `POST /api/rephrase`

**Request:**
```json
{
  "text": "Your toxic comment here",
  "num_suggestions": 3,
  "include_comparison": true
}
```

**Response:**
```json
{
  "original_text": "Your toxic comment here",
  "suggestions": [
    {
      "text": "Rephrased version",
      "strategy": "Word Replacement",
      "changes": ["'hate' → 'dislike'"],
      "description": "Replaced offensive words with neutral alternatives",
      "comparison": {
        "original": {"overall": 0.85, "level": "High"},
        "rephrased": {"overall": 0.35, "level": "Medium"},
        "improvement": 0.50,
        "improvement_percent": 58.8,
        "is_better": true
      }
    }
  ],
  "count": 3
}
```

### Comparison API

**Endpoint:** `POST /api/compare`

**Request:**
```json
{
  "original": "Original text",
  "rephrased": "Rephrased text"
}
```

**Response:**
```json
{
  "original": {
    "scores": {...},
    "overall": 0.85,
    "level": "High"
  },
  "rephrased": {
    "scores": {...},
    "overall": 0.35,
    "level": "Medium"
  },
  "improvement": 0.50,
  "improvement_percent": 58.8,
  "is_better": true
}
```

## Training the Model

To train or retrain the model:

```bash
python train_model.py
```

**Training features:**
- Uses Civil Comments dataset
- 20,000 samples (configurable)
- 90/10 train/validation split
- Evaluation metrics (F1, Accuracy)
- Checkpointing and best model selection

## Project Structure

```
toxic_app/
├── app.py                 # Flask application with routes
├── config.py             # Configuration management
├── utils.py              # Model utilities and analysis functions
├── train_model.py        # Model training script
├── requirements.txt      # Python dependencies
├── static/
│   └── style.css         # UI styling
├── templates/
│   └── index.html        # Web interface
└── bert_toxic_model/     # Trained model files
```

## Features Overview

### Web Interface
- Character counter with visual feedback
- Clear button for quick reset
- Loading spinner during analysis
- Copy to clipboard functionality
- Responsive design
- Error messages and validation

### Analysis Features
- 6 toxicity categories (toxicity, severe_toxicity, obscene, threat, insult, identity_attack)
- Overall toxicity score and level (Low/Medium/High)
- Sentiment analysis (Positive/Negative/Neutral)
- Visual word highlighting
- Interactive pie chart distribution
- Detailed explanations

### Rephrase Features
- **4 Rephrasing Strategies:**
  - Word Replacement - Swap toxic words with neutral alternatives
  - Expression Rewrite - Restructure aggressive phrases
  - Tone Softening - Add polite qualifiers and reduce emphasis
  - Question Format - Convert statements to constructive questions
- Before/After toxicity comparison
- Real-time improvement percentage
- One-click apply to text field
- Copy suggestions to clipboard

### Technical Features
- Model lazy loading and caching
- Input validation (1-500 characters)
- Rate limiting protection
- Comprehensive error handling
- Structured logging
- REST API support
- Health check endpoint (`/health`)

## Security

- Rate limiting prevents abuse
- Input validation prevents malicious data
- Configurable secret key for production
- Debug mode disabled by default
- Comprehensive error logging

## Performance

- Model loaded once and cached in memory
- Long text handling (100+ word optimization)
- Efficient tokenization
- Background processing optimization

## Troubleshooting

**Model not found:**
```bash
python train_model.py
```

**Port already in use:**
```bash
# Change port in .env or:
PORT=5002 python app.py
```

**Rate limit errors:**
- Wait 1 minute between requests
- Reduce request frequency

**Dependencies issues:**
```bash
pip install --upgrade -r requirements.txt
```

## License

MIT License

## Contributing

Feel free to submit issues and enhancement requests!
