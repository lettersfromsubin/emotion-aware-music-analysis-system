# lyrics-emotion-analysis-assistant

A simple Gradio app for finding song lyrics in the public Hugging Face dataset `theelderemo/genius-lyrics-cleaned` and generating an emotion-focused analysis with Gemini.

The app no longer depends on live Genius retrieval as its main workflow. That change makes the project more stable because live Genius scraping can be blocked by Cloudflare. Lyrics are used only at runtime for educational and research-style analysis.

## Tech Stack

- Python
- Hugging Face Datasets
- Gemini API (`google-genai`)
- FAISS (`faiss-cpu`)
- scikit-learn
- matplotlib
- Sentence Transformers
- Gradio
- python-dotenv
- requests
- iTunes Search API
- Spotify Tracks Dataset on Hugging Face

## Features

- Stream-search the public dataset `theelderemo/genius-lyrics-cleaned`
- Avoid downloading the full dataset up front by using `streaming=True`
- Match songs by artist and title with case-insensitive search
- Return title, artist, year, tag, and lyrics when a match is found
- Fall back to manual lyrics paste when the song is not found in the initial streamed window
- Use `GEMINI_API_KEY` for structured lyrics analysis
- Include emotion confidence scoring and a short justification for the selected emotion
- Restrict dominant emotion labels to a fixed portfolio-friendly taxonomy
- Fall back to a simple rule-based analysis if Gemini is unavailable
- Optionally show album artwork, album/collection name, and an iTunes track link from the free iTunes Search API
- Add Spotify-style track metadata from the public Hugging Face dataset `maharshipandya/spotify-tracks-dataset`
- Use FAISS vector search for faster embedding-based similar song recommendations, with cosine similarity as fallback
- Assign a lightweight valence-arousal mood cluster such as sad/calm, energetic/joyful, warm/peaceful, tense/angry, or mixed/reflective
- Show a simple valence-arousal emotion-space map for the analyzed song

## Project Structure

```text
lyrics-emotion-analysis-assistant/
├── app.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
├── screenshots/
└── vectorstore/
```

## Setup

```bash
cd lyrics-emotion-analysis-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create or update `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
HF_DATASET_SEARCH_LIMIT=50000
```

## How It Works

1. The app streams rows from the public Hugging Face dataset `theelderemo/genius-lyrics-cleaned`.
2. It searches for a case-insensitive artist/title match without downloading the full dataset at startup.
3. It optionally looks up album artwork metadata through the free iTunes Search API.
4. It optionally matches the song against `maharshipandya/spotify-tracks-dataset` for popularity, album, and duration metadata.
5. If a match is found, the lyrics are analyzed in memory.
6. If a match is not found in the initial streamed window, you can paste the lyrics manually and analyze them directly.

## Spotify Dataset Metadata

The app does not use the Spotify API. It loads the public Hugging Face dataset `maharshipandya/spotify-tracks-dataset` and matches by `track_name` and `artists` with case-insensitive search. When available, the output shows popularity, album, duration, and a simple popularity interpretation.

## Optional Album Artwork

The app uses the public iTunes Search API to fetch album artwork and basic track metadata. This does not require authentication and does not use the Spotify API. If iTunes does not return a result, the analysis still works without artwork.

## Why Live Genius Retrieval Was Removed

Earlier versions used live Genius retrieval, but that approach was unreliable because Genius can block automated requests with Cloudflare. This refactor makes the app more stable and easier to demo in a portfolio setting by using a public dataset as the primary source instead.

## Runtime-Only Lyrics Usage

- Lyrics are used only during runtime for educational and research analysis.
- The app does not save lyrics to files or a database.
- Manual lyrics pasted into the UI are processed in memory for the current request only.

## Run the App

```bash
python3 app.py
```

Then open the local Gradio URL shown in the terminal.
