AI Sentiment Radar — VibeCheck
VibeCheck is a real-time NLP dashboard that scans the global news landscape to quantify the "vibe" of any given topic. By pulling live headlines from Google News and processing them through Sentiment Analysis, it provides a visual breakdown of public perception and media bias.

✨ Features
Live News Integration: Fetches real-time headlines using Google News RSS feeds.

NLP Analysis: Utilizes TextBlob to calculate Polarity (positive/negative) and Subjectivity (opinion vs. fact) for every headline.

Dynamic Visualizations:

Sentiment Donut: A high-level breakdown of Positive, Negative, and Neutral coverage.

Polarity Bar Chart: Granular view of the most impactful headlines.

Vibe-Meter: A custom-styled CSS visualizer for individual article sentiment.

Source Intelligence: Identifies which news outlets are reporting on the topic and their average sentiment lean.

Modern UI: A custom dark-themed interface built with Inter fonts, glassmorphism effects, and responsive CSS.

🛠️ Tech Stack
Frontend: Streamlit

NLP Engine: TextBlob

Data Handling: Pandas

Visualization: Matplotlib

API/Data: Google News RSS via requests and xml.etree

🚀 Getting Started
1. Install dependencies
Ensure you have Python 3.8+ installed, then run:

Bash
pip install streamlit pandas requests textblob matplotlib
2. Download NLP Corpora
TextBlob requires a one-time download for its underlying models:

Bash
python -m textblob.download_corpora
3. Run the App
Save the code as app.py and run:

Bash
streamlit run app.py
📊 How it Works
The application follows a three-step pipeline:

Ingestion: The app sanitizes your search query and requests an XML feed from Google News.

Processing: It parses the XML, cleans HTML artifacts, and passes the text to a TextBlob object.

Scoring: * Polarity > 0.1: Positive 😊

Polarity < -0.1: Negative 😠

In-between: Neutral ⚪

💡 Example Queries
Try these in the search bar for interesting results:

Bitcoin (High volatility, mixed sentiment)

Artificial Intelligence (Highly subjective/polarizing)

James Webb Space Telescope (Usually high positive polarity)