## Setup Instructions

### 1. Clone the repository
git clone https://github.com/anubhav-goyal/youtube-sentiment
cd youtube-sentiment

### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

### 3. Install dependencies
pip install -r requirements.txt

### 4. Set up environment variables
Create a .env file in project root:
YOUTUBE_API_KEY=your_youtube_api_key_here

Get YouTube API key from:
console.cloud.google.com → Enable YouTube Data API v3

### 5. Run the API
uvicorn api.main:app --reload --port 8000

Model downloads automatically from HuggingFace Hub
on first run (~250MB, takes 2-3 minutes)

### 6. Run the Streamlit app
streamlit run app/streamlit_app.py

### 7. Or use Docker
docker build -t sentiment-api .
docker run -p 8000:8000 --env-file .env sentiment-api