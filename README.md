# 🎧 Emotion-Aware Music Analysis System

An AI-powered system for analyzing song lyrics using LLMs and mapping them into a structured emotion space (valence–arousal), enabling intelligent music understanding, interpretation, and recommendation.

---

### Role
- AI System Architecture Design
- LLM Pipeline Development
- Emotion Modeling & Calibration Layer Design
- Recommendation System Development (FAISS)
- Backend Integration & Deployment

### Contribution
100% (Individual Project)

---

## 🚀 Live Demo

👉 Hugging Face Space:  
https://huggingface.co/spaces/ALEXJK0901/emotion-aware-music-analysis-system

---

## 🚀 Key Highlights

* LLM-based emotion analysis using **Gemini**
* Hybrid affective modeling (**semantic emotion + numeric space + consistency layer**)
* Emotion-to-valence/arousal mapping (psychological model)
* Post-processing calibration for **affective consistency**
* FAISS-powered similarity search for recommendations
* Emotion-space visualization (valence–arousal map)
* Multi-source integration (lyrics + Spotify-style metadata + iTunes)

---

## 🧠 System Architecture

### 1. Lyrics Retrieval
* Hugging Face dataset streaming (`theelderemo/genius-lyrics-cleaned`)
* No full dataset download required

### 2. Emotion Analysis (LLM)
* Gemini API
* Structured output:
  * Emotion category
  * Emotion nuance
  * Confidence score
  * Raw valence & arousal

### 3. Affective Calibration Layer (Key Contribution)

A lightweight post-processing module ensures **structural consistency between emotion category and numeric representation**.

Includes:
* Emotion-to-valence/arousal mapping
* Adaptive blending between model output and mapping
* Low-arousal emotion constraints (e.g., longing, nostalgia, melancholy)
* Cluster override rules for edge cases

This prevents inconsistencies such as:
> “longing” being classified as “tense/angry”

### 4. Emotion Modeling

Hybrid pipeline:
* LLM → semantic emotion understanding
* Mapping → numeric representation
* Calibration → consistency correction
* Rules → stability and interpretability

### 5. Mood Clustering

Valence–arousal space is mapped into:
* sad / calm  
* energetic / joyful  
* peaceful / warm  
* tense / angry  
* mixed / reflective  

### 6. Recommendation Engine

* Sentence Transformers embeddings
* FAISS vector search
* Cosine similarity fallback

### 7. Visualization

* Emotion-space plotting (valence vs arousal)
* Real-time position of the analyzed song

---

## 🎯 Use Cases

* Emotion-aware music recommendation systems
* AI-powered content analysis for entertainment platforms
* Playlist generation based on emotional context
* Creative support for music production
* Decision-support systems for media platforms 
---

## 🛠 Tech Stack

* Python
* Hugging Face Datasets
* Gemini API (google-genai)
* FAISS (faiss-cpu)
* Sentence Transformers
* scikit-learn
* matplotlib
* Gradio
* python-dotenv
* requests
* iTunes Search API
* Spotify Tracks Dataset (Hugging Face)

---

## ⚙️ Features

* Stream-search lyrics dataset using `streaming=True`
* Case-insensitive artist/title matching
* Manual lyrics fallback
* Structured LLM-based emotion analysis
* Emotion confidence + justification
* Controlled emotion taxonomy
* Affective calibration for consistency
* Rule-based fallback if LLM unavailable
* Spotify-style metadata enrichment
* iTunes artwork integration
* FAISS-based similar song recommendations
* Emotion-space visualization

---

## 📂 Project Structure

```
lyrics-emotion-analysis-assistant/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── screenshots/
└── vectorstore/
```

---

## 🔧 Setup

```bash
cd lyrics-emotion-analysis-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
HF_DATASET_SEARCH_LIMIT=50000
```

---

## ▶️ Run the App

```bash
python3 app.py
```

Open the local Gradio URL shown in the terminal.

---

## 🔍 How It Works

* Streams data from Hugging Face dataset (no full download)
* Matches songs by artist/title
* Enriches metadata via:
  * iTunes API (artwork, album)
  * Spotify dataset (popularity, duration)
* Performs real-time LLM analysis
* Applies calibration layer for consistency
* Generates structured emotional insights
* Finds similar songs via embeddings
* Visualizes emotion in valence–arousal space

---

## 🎵 Spotify Dataset Integration

* Dataset: `maharshipandya/spotify-tracks-dataset`
* No Spotify API required
* Matching via track name + artist
* Outputs:
  * popularity
  * album
  * duration
  * interpretation

---

## 🖼 Optional Album Artwork

* Uses iTunes Search API (no authentication)
* Fetches album artwork and metadata
* Fully optional

---

## ⚠️ Design Decision: No Live Genius Scraping

Removed due to:
* Cloudflare blocking
* instability

Replaced with:
* public dataset → reproducibility and stability

---

## 🔒 Runtime-Only Lyrics Usage

* Lyrics processed only in memory
* No storage or persistence
* Manual input not saved
* Designed for research/educational use

---

## 📊 Future Improvements

* Probabilistic emotion mapping (soft consistency)
* Full dataset indexing
* Multilingual support
* Advanced clustering
* Real-time API integrations

---


Developed as part of an LLM-based content intelligence system for emotion-aware analysis and recommendation.
