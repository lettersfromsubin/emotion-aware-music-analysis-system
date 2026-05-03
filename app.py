from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

from dotenv import load_dotenv

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    import gradio as gr
except ImportError:  # pragma: no cover
    gr = None

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from sklearn.cluster import KMeans
except ImportError:  # pragma: no cover
    KMeans = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None


APP_TITLE = "Emotion-Aware Music Analysis System"
ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"
DATASET_NAME = "theelderemo/genius-lyrics-cleaned"
SPOTIFY_DATASET_NAME = "maharshipandya/spotify-tracks-dataset"
ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
SIMILARITY_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_DATASET_SEARCH_LIMIT = 50_000
_spotify_dataset: Any | None = None
_similarity_model: Any | None = None
_song_embedding_matrix: Any | None = None
_normalized_song_embedding_matrix: Any | None = None
_faiss_index: Any | None = None
_faiss_disabled = False
_faiss_startup_attempted = False
_mood_cluster_model: Any | None = None

load_dotenv(ENV_PATH, override=False)

MISSING_DISPLAY_VALUES = {
    "",
    "n/a",
    "na",
    "none",
    "null",
    "nan",
    "not found",
    "unknown",
}

SPOTIFY_CSS = """
:root {
    --app-bg: #121212;
    --panel-bg: #181818;
    --panel-soft: #202020;
    --panel-border: rgba(255, 255, 255, 0.08);
    --text-primary: #ffffff;
    --text-secondary: #b3b3b3;
    --accent: #1db954;
    --accent-soft: rgba(29, 185, 84, 0.16);
    --purple: #8b5cf6;
}

.gradio-container {
    background: radial-gradient(circle at top left, rgba(29, 185, 84, 0.13), transparent 34%),
        var(--app-bg) !important;
    color: var(--text-primary) !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.app-shell {
    max-width: 1180px;
    margin: 0 auto;
    padding: 28px 18px 44px;
}

.hero {
    padding: 28px;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(29, 185, 84, 0.18), rgba(139, 92, 246, 0.13)),
        #181818;
    border: 1px solid var(--panel-border);
    box-shadow: 0 18px 60px rgba(0, 0, 0, 0.36);
}

.hero h1 {
    margin: 0 0 10px;
    font-size: clamp(2rem, 4vw, 3.8rem);
    line-height: 1;
    letter-spacing: 0;
    color: var(--text-primary);
}

.hero p {
    margin: 0;
    color: var(--text-secondary);
    font-size: 1.04rem;
    line-height: 1.6;
    max-width: 760px;
}

.section-title h2,
.section-title h3 {
    color: var(--text-primary);
    margin-bottom: 8px;
}

.input-card,
.track-card-shell,
.metric-card,
.plot-card,
.similar-card,
.details-card {
    background: rgba(24, 24, 24, 0.96);
    border: 1px solid var(--panel-border);
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 14px 40px rgba(0, 0, 0, 0.22);
}

.input-card textarea,
.input-card input {
    background: #0f0f0f !important;
    color: var(--text-primary) !important;
    border-color: rgba(255, 255, 255, 0.12) !important;
}

.input-card label,
.metric-card label,
.plot-card label {
    color: var(--text-secondary) !important;
}

.analyze-button {
    border-radius: 999px !important;
    background: var(--accent) !important;
    color: #06170c !important;
    font-weight: 800 !important;
    border: 0 !important;
    min-height: 48px;
}

.status-text {
    color: var(--text-secondary);
    min-height: 32px;
}

.track-card {
    display: grid;
    grid-template-columns: minmax(180px, 260px) minmax(0, 1fr);
    gap: 26px;
    align-items: center;
}

.cover-frame {
    width: 100%;
    aspect-ratio: 1;
    border-radius: 20px;
    overflow: hidden;
    background: linear-gradient(135deg, #2a2a2a, #101010);
    box-shadow: 0 22px 54px rgba(0, 0, 0, 0.44);
}

.cover-frame img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}

.cover-placeholder {
    width: 100%;
    height: 100%;
    display: grid;
    place-items: center;
    color: rgba(255, 255, 255, 0.68);
    font-size: 3rem;
}

.track-kicker {
    color: var(--accent);
    font-weight: 800;
    text-transform: uppercase;
    font-size: 0.76rem;
    letter-spacing: 0.12em;
    margin-bottom: 10px;
}

.track-title {
    color: var(--text-primary);
    font-size: clamp(2rem, 5vw, 4.8rem);
    line-height: 0.96;
    font-weight: 900;
    letter-spacing: 0;
    margin-bottom: 12px;
    overflow-wrap: anywhere;
}

.track-artist {
    color: var(--text-secondary);
    font-size: 1.08rem;
    margin-bottom: 20px;
}

.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 12px 0 18px;
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    border-radius: 999px;
    padding: 8px 12px;
    background: var(--panel-soft);
    color: var(--text-primary);
    border: 1px solid rgba(255, 255, 255, 0.08);
    font-weight: 700;
}

.pill.accent {
    background: var(--accent-soft);
    color: #b9ffd0;
    border-color: rgba(29, 185, 84, 0.26);
}

.pill.purple {
    background: rgba(139, 92, 246, 0.16);
    color: #ddd0ff;
    border-color: rgba(139, 92, 246, 0.28);
}

.track-meta {
    color: var(--text-secondary);
    display: flex;
    flex-wrap: wrap;
    gap: 10px 16px;
    font-size: 0.94rem;
}

.track-meta span {
    background: rgba(255, 255, 255, 0.055);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 999px;
    padding: 7px 10px;
}

.track-meta a {
    color: #b9ffd0;
    text-decoration: none;
    font-weight: 700;
}

.metadata-note {
    margin-top: 14px;
    color: var(--text-secondary);
    font-size: 0.92rem;
}

.slider-wrap .wrap {
    background: transparent !important;
}

.plot-card img {
    border-radius: 18px !important;
}

.similar-card,
.details-card {
    color: var(--text-primary);
}

.similar-card h3,
.similar-card li,
.details-card p,
.details-card li {
    color: var(--text-primary);
}

.details-card {
    color: var(--text-secondary);
}

@media (max-width: 760px) {
    .track-card {
        grid-template-columns: 1fr;
    }

    .cover-frame {
        max-width: 280px;
    }
}
"""


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "us",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}

POSITIVE_WORDS = {
    "alive",
    "beautiful",
    "bright",
    "celebrate",
    "dream",
    "free",
    "good",
    "happy",
    "heaven",
    "hope",
    "joy",
    "kiss",
    "light",
    "love",
    "peace",
    "shine",
    "smile",
    "strong",
    "sweet",
    "win",
}

NEGATIVE_WORDS = {
    "alone",
    "angry",
    "broken",
    "cold",
    "cry",
    "dark",
    "dead",
    "fear",
    "fight",
    "gone",
    "hate",
    "heartbreak",
    "hurt",
    "lost",
    "pain",
    "regret",
    "sad",
    "shadow",
    "tears",
    "war",
}

HIGH_AROUSAL_WORDS = {
    "burn",
    "dance",
    "fire",
    "fight",
    "fly",
    "loud",
    "run",
    "scream",
    "shake",
    "tonight",
    "wild",
}

ALLOWED_EMOTIONS = (
    "joy",
    "sadness",
    "nostalgia",
    "longing",
    "anger",
    "confidence",
    "melancholy",
    "hope",
    "love",
    "tension",
    "desire",
    "regret",
    "loneliness",
    "freedom",
    "fear",
)

EMOTION_LEXICONS = {
    "joy": {"celebrate", "dance", "free", "happy", "joy", "light", "smile", "sun"},
    "sadness": {"alone", "broken", "cry", "gone", "hurt", "sad", "tears"},
    "nostalgia": {"before", "memory", "old", "remember", "used", "yesterday"},
    "longing": {"away", "back", "dream", "miss", "need", "want", "wish"},
    "anger": {"angry", "blood", "fight", "hate", "rage", "war"},
    "confidence": {"best", "boss", "power", "strong", "top", "win"},
    "melancholy": {"blue", "cold", "dark", "empty", "lonely", "rain", "shadow"},
    "hope": {"again", "believe", "breathe", "heal", "hope", "rise", "tomorrow"},
    "love": {"baby", "heart", "kiss", "love", "touch", "together"},
    "tension": {"afraid", "burn", "fear", "fight", "fire", "ghost", "nightmare", "scream"},
    "desire": {"body", "crave", "desire", "need", "touch", "want"},
    "regret": {"apologize", "fault", "mistake", "regret", "sorry", "wrong"},
    "loneliness": {"alone", "empty", "lonely", "nobody", "solitude"},
    "freedom": {"escape", "fly", "free", "open", "road", "run", "sky"},
    "fear": {"afraid", "dark", "fear", "ghost", "nightmare", "shadow"},
}

THEME_LEXICONS = {
    "love and connection": {"baby", "heart", "hold", "kiss", "love", "touch", "together"},
    "heartbreak and loss": {"alone", "broken", "bye", "cry", "gone", "leave", "lost"},
    "hope and resilience": {"again", "breathe", "dream", "heal", "hope", "rise", "strong"},
    "identity and self-reflection": {"change", "find", "know", "mirror", "myself", "name", "who"},
    "escape and freedom": {"fly", "free", "road", "run", "sky", "tonight"},
    "conflict and struggle": {"battle", "fight", "fire", "pain", "war"},
    "nostalgia and memory": {"before", "memory", "remember", "yesterday"},
}

MOOD_CLUSTER_ANCHORS = (
    {
        "label": "sad or calm",
        "center": (0.2, 0.2),
        "explanation": "Low valence and low arousal suggest a subdued mood, often read as sad, calm, or emotionally quiet.",
    },
    {
        "label": "energetic or joyful",
        "center": (0.8, 0.8),
        "explanation": "High valence and high arousal point to bright, active emotion with an energetic or joyful feel.",
    },
    {
        "label": "warm or peaceful",
        "center": (0.8, 0.2),
        "explanation": "High valence with low arousal suggests a positive but gentle mood, closer to warmth or peace than excitement.",
    },
    {
        "label": "tense or angry",
        "center": (0.2, 0.8),
        "explanation": "Low valence with high arousal indicates an intense negative mood, often experienced as tension, anger, or unrest.",
    },
    {
        "label": "mixed or reflective",
        "center": (0.5, 0.5),
        "explanation": "Mid-range valence and arousal suggest emotional ambiguity, balance, or a reflective mixed mood.",
    },
)

SIMILAR_SONG_CATALOG = [
    {
        "title": "Happy",
        "artist": "Pharrell Williams",
        "text": "joy bright upbeat celebration confidence dance positive high valence high arousal",
    },
    {
        "title": "Good as Hell",
        "artist": "Lizzo",
        "text": "joy confidence empowerment self love upbeat pop celebration high energy",
    },
    {
        "title": "Someone Like You",
        "artist": "Adele",
        "text": "sadness regret longing heartbreak farewell lost love low valence ballad",
    },
    {
        "title": "The Night We Met",
        "artist": "Lord Huron",
        "text": "sadness nostalgia regret longing memory loss melancholy reflective low arousal",
    },
    {
        "title": "Ribs",
        "artist": "Lorde",
        "text": "nostalgia youth memory bittersweet growing up longing melancholy indie pop",
    },
    {
        "title": "All I Want",
        "artist": "Kodaline",
        "text": "longing heartbreak desire sadness intimate vulnerable romantic ballad",
    },
    {
        "title": "You Oughta Know",
        "artist": "Alanis Morissette",
        "text": "anger betrayal resentment high arousal breakup emotional release rock",
    },
    {
        "title": "Stronger",
        "artist": "Kanye West",
        "text": "confidence power resilience bold energy victory assertive high arousal",
    },
    {
        "title": "Holocene",
        "artist": "Bon Iver",
        "text": "melancholy quiet reflection nature distance low arousal introspective indie folk",
    },
    {
        "title": "Fix You",
        "artist": "Coldplay",
        "text": "hope healing sadness comfort resilience gradual uplift emotional repair",
    },
    {
        "title": "All of Me",
        "artist": "John Legend",
        "text": "love devotion intimacy romance vulnerability positive ballad",
    },
    {
        "title": "Take Me to Church",
        "artist": "Hozier",
        "text": "tension desire love conflict dramatic intensity soul rock",
    },
    {
        "title": "Earned It",
        "artist": "The Weeknd",
        "text": "desire sensual intimacy slow burn romance dark pop",
    },
    {
        "title": "Back to December",
        "artist": "Taylor Swift",
        "text": "regret apology nostalgia lost love longing reflective country pop",
    },
    {
        "title": "Dancing On My Own",
        "artist": "Robyn",
        "text": "loneliness heartbreak dance pop longing isolation bittersweet high arousal",
    },
    {
        "title": "Dog Days Are Over",
        "artist": "Florence + The Machine",
        "text": "freedom release joy escape movement catharsis high energy",
    },
    {
        "title": "Bury a Friend",
        "artist": "Billie Eilish",
        "text": "fear darkness tension eerie anxiety nightmare low valence electronic pop",
    },
]


def get_env_value(name: str) -> str:
    load_dotenv(ENV_PATH, override=False)
    return os.getenv(name, "").strip()


def require_dependency(module: Any, package_name: str) -> None:
    if module is None:
        raise RuntimeError(
            f"Missing dependency '{package_name}'. Install project requirements before running the app."
        )


def clean_lyrics(raw_lyrics: str) -> str:
    lyrics = raw_lyrics.replace("\r\n", "\n").strip()
    lyrics = re.sub(r"\n?\d*Embed\s*$", "", lyrics).strip()
    lyrics = re.sub(r"[ \t]+\n", "\n", lyrics)
    lyrics = re.sub(r"\n{3,}", "\n\n", lyrics)
    return lyrics


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def safe_display(value: Any, fallback: str = "Unknown") -> str:
    if value is None:
        return fallback

    if isinstance(value, float) and value != value:
        return fallback

    text = str(value).strip()
    if text.lower() in MISSING_DISPLAY_VALUES:
        return fallback

    return text


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def get_mood_cluster_model() -> Any:
    require_dependency(KMeans, "scikit-learn")
    require_dependency(np, "numpy")

    global _mood_cluster_model
    if _mood_cluster_model is None:
        anchors = np.asarray([cluster["center"] for cluster in MOOD_CLUSTER_ANCHORS], dtype="float32")
        _mood_cluster_model = KMeans(
            n_clusters=len(MOOD_CLUSTER_ANCHORS),
            init=anchors,
            n_init=1,
            max_iter=1,
            random_state=42,
        ).fit(anchors)
    return _mood_cluster_model


def get_nearest_mood_anchor(valence: float, arousal: float) -> dict[str, Any]:
    return min(
        MOOD_CLUSTER_ANCHORS,
        key=lambda cluster: (valence - cluster["center"][0]) ** 2 + (arousal - cluster["center"][1]) ** 2,
    )


def get_mood_anchor_for_kmeans_cluster(model: Any, cluster_index: int) -> dict[str, Any]:
    anchor_centers = np.asarray([cluster["center"] for cluster in MOOD_CLUSTER_ANCHORS], dtype="float32")
    cluster_center = model.cluster_centers_[cluster_index]
    anchor_index = int(np.argmin(np.linalg.norm(anchor_centers - cluster_center, axis=1)))
    return MOOD_CLUSTER_ANCHORS[anchor_index]


def assign_mood_cluster(valence: Any, arousal: Any) -> dict[str, str]:
    try:
        normalized_valence = clamp(float(valence), 0.0, 1.0)
        normalized_arousal = clamp(float(arousal), 0.0, 1.0)
    except (TypeError, ValueError):
        return {
            "cluster_label": "mixed or reflective",
            "explanation": (
                "Valence or arousal was unavailable, so the song is assigned to the mid-range reflective cluster."
            ),
        }

    try:
        model = get_mood_cluster_model()
        point = np.asarray([[normalized_valence, normalized_arousal]], dtype="float32")
        cluster_index = int(model.predict(point)[0])
        cluster = get_mood_anchor_for_kmeans_cluster(model, cluster_index)
    except Exception:
        cluster = get_nearest_mood_anchor(normalized_valence, normalized_arousal)

    return {
        "cluster_label": cluster["label"],
        "explanation": (
            f"{cluster['explanation']} Detected valence: {normalized_valence:.2f}; "
            f"arousal: {normalized_arousal:.2f}."
        ),
    }


def create_valence_arousal_plot(valence: Any, arousal: Any, song_title: str) -> str | None:
    require_dependency(plt, "matplotlib")

    try:
        normalized_valence = clamp(float(valence), 0.0, 1.0)
        normalized_arousal = clamp(float(arousal), 0.0, 1.0)
    except (TypeError, ValueError):
        return None

    display_title = str(song_title).strip() or "Analyzed song"
    point_label = display_title if len(display_title) <= 36 else f"{display_title[:33]}..."

    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=140)
    try:
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#f8fafc")

        ax.fill_between([0.0, 0.5], 0.0, 0.5, color="#dbeafe", alpha=0.55)
        ax.fill_between([0.5, 1.0], 0.0, 0.5, color="#dcfce7", alpha=0.55)
        ax.fill_between([0.0, 0.5], 0.5, 1.0, color="#fee2e2", alpha=0.55)
        ax.fill_between([0.5, 1.0], 0.5, 1.0, color="#fef3c7", alpha=0.6)

        ax.axvline(0.5, color="#64748b", linewidth=1.1, alpha=0.75)
        ax.axhline(0.5, color="#64748b", linewidth=1.1, alpha=0.75)
        ax.scatter(
            [normalized_valence],
            [normalized_arousal],
            s=120,
            color="#2563eb",
            edgecolor="white",
            linewidth=1.6,
            zorder=4,
        )
        ax.annotate(
            point_label,
            (normalized_valence, normalized_arousal),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color="#0f172a",
            zorder=5,
        )

        quadrant_labels = [
            (0.04, 0.04, "sad/calm", "left", "bottom"),
            (0.96, 0.96, "joyful/energetic", "right", "top"),
            (0.96, 0.04, "peaceful/warm", "right", "bottom"),
            (0.04, 0.96, "tense/angry", "left", "top"),
        ]
        for x, y, label, horizontal_alignment, vertical_alignment in quadrant_labels:
            ax.text(
                x,
                y,
                label,
                transform=ax.transAxes,
                ha=horizontal_alignment,
                va=vertical_alignment,
                fontsize=10,
                color="#334155",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Valence (0 = negative, 1 = positive)")
        ax.set_ylabel("Arousal (0 = calm, 1 = energetic)")
        ax.set_title(f"{display_title} in Emotion Space", pad=12)
        ax.grid(True, color="#cbd5e1", linewidth=0.8, alpha=0.55)

        output_file = tempfile.NamedTemporaryFile(
            prefix="valence_arousal_",
            suffix=".png",
            delete=False,
        )
        output_path = output_file.name
        output_file.close()
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
        return output_path
    finally:
        plt.close(fig)


def normalize_emotion_label(value: Any) -> str:
    emotion = str(value).strip().lower()
    if emotion in ALLOWED_EMOTIONS:
        return emotion
    for allowed_emotion in ALLOWED_EMOTIONS:
        if allowed_emotion in emotion:
            return allowed_emotion
    return "melancholy"


def normalize_sentiment_label(value: Any) -> str:
    sentiment = str(value).strip().lower()
    if sentiment in {"positive", "negative", "neutral"}:
        return sentiment
    if sentiment in {"mixed", "ambivalent", "bittersweet"}:
        return "neutral"
    if "positive" in sentiment:
        return "positive"
    if "negative" in sentiment:
        return "negative"
    return "neutral"


def get_dataset_search_limit() -> int:
    raw_value = get_env_value("HF_DATASET_SEARCH_LIMIT")
    if not raw_value:
        return DEFAULT_DATASET_SEARCH_LIMIT
    try:
        return max(1, int(raw_value))
    except ValueError:
        return DEFAULT_DATASET_SEARCH_LIMIT


def get_similarity_model() -> Any:
    require_dependency(SentenceTransformer, "sentence-transformers")

    global _similarity_model
    if _similarity_model is None:
        _similarity_model = SentenceTransformer(SIMILARITY_MODEL_NAME)
    return _similarity_model


def normalize_embedding_matrix(embeddings: Any) -> Any:
    require_dependency(np, "numpy")

    matrix = np.asarray(embeddings, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Embeddings must be a non-empty 2D matrix.")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normalized = matrix / norms
    return np.ascontiguousarray(normalized, dtype="float32")


def build_faiss_index(embeddings: Any) -> Any:
    require_dependency(faiss, "faiss-cpu")

    normalized_embeddings = normalize_embedding_matrix(embeddings)
    index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
    index.add(normalized_embeddings)
    return index


def get_song_embedding_matrix() -> Any:
    global _song_embedding_matrix
    if _song_embedding_matrix is None:
        model = get_similarity_model()
        texts = [song["text"] for song in SIMILAR_SONG_CATALOG]
        _song_embedding_matrix = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return _song_embedding_matrix


def get_normalized_song_embedding_matrix() -> Any:
    global _normalized_song_embedding_matrix
    if _normalized_song_embedding_matrix is None:
        _normalized_song_embedding_matrix = normalize_embedding_matrix(get_song_embedding_matrix())
    return _normalized_song_embedding_matrix


def initialize_faiss_recommendations() -> None:
    global _faiss_disabled, _faiss_index, _faiss_startup_attempted
    if _faiss_startup_attempted:
        return

    _faiss_startup_attempted = True
    try:
        _faiss_index = build_faiss_index(get_song_embedding_matrix())
    except Exception:
        _faiss_index = None
        _faiss_disabled = True


def format_similarity_results(indices: list[int], scores: list[float]) -> list[dict[str, Any]]:
    results = []
    for index, score in zip(indices, scores):
        if index < 0 or index >= len(SIMILAR_SONG_CATALOG):
            continue
        song = SIMILAR_SONG_CATALOG[index]
        results.append(
            {
                "title": song["title"],
                "artist": song["artist"],
                "similarity_score": round(float(score), 4),
            }
        )
    return results


def get_similar_songs_cosine(query_lyrics: str, top_k: int = 5) -> list[dict[str, Any]]:
    if not query_lyrics.strip() or top_k <= 0:
        return []

    try:
        model = get_similarity_model()
        song_embeddings = get_normalized_song_embedding_matrix()
        query_embedding = model.encode([query_lyrics], convert_to_numpy=True, show_progress_bar=False)
        normalized_query = normalize_embedding_matrix(query_embedding)
        scores = song_embeddings @ normalized_query[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return format_similarity_results(top_indices.tolist(), scores[top_indices].tolist())
    except Exception:
        return []


def get_similar_songs_faiss(query_lyrics: str, top_k: int = 5) -> list[dict[str, Any]]:
    if not query_lyrics.strip() or top_k <= 0:
        return []

    try:
        global _faiss_disabled, _faiss_index
        if _faiss_disabled:
            return get_similar_songs_cosine(query_lyrics, top_k=top_k)

        if _faiss_index is None:
            _faiss_index = build_faiss_index(get_song_embedding_matrix())

        model = get_similarity_model()
        query_embedding = model.encode([query_lyrics], convert_to_numpy=True, show_progress_bar=False)
        normalized_query = normalize_embedding_matrix(query_embedding)
        scores, indices = _faiss_index.search(normalized_query, min(top_k, len(SIMILAR_SONG_CATALOG)))
        return format_similarity_results(indices[0].tolist(), scores[0].tolist())
    except Exception:
        _faiss_disabled = True
        return get_similar_songs_cosine(query_lyrics, top_k=top_k)


def format_embedding_recommendations(similar_songs: list[dict[str, Any]]) -> list[str]:
    return [
        (
            f"{song['title']} — {song['artist']}: lyric embedding similarity "
            f"{song['similarity_score']:.2f}."
        )
        for song in similar_songs
    ]


def build_dataset_song_data(row: dict[str, Any]) -> dict[str, str]:
    return {
        "title": safe_display(row.get("title"), "Untitled song"),
        "artist": safe_display(row.get("artist"), "Unknown artist"),
        "year": safe_display(row.get("year"), "Unknown"),
        "tag": safe_display(row.get("tag"), "Unknown"),
        "lyrics": clean_lyrics(str(row.get("lyrics", ""))),
        "source": f"Hugging Face dataset ({DATASET_NAME})",
    }


def build_dataset_not_found_result(artist: str, title: str, max_rows: int) -> dict[str, str]:
    message = (
        f"No case-insensitive dataset match found for '{artist} - {title}' within the first "
        f"{max_rows:,} streamed rows of {DATASET_NAME}. Paste lyrics into Manual Lyrics Paste "
        "and click Analyze again."
    )
    return {
        "title": safe_display(title, "Untitled song"),
        "artist": safe_display(artist, "Unknown artist"),
        "year": "Unknown",
        "tag": "not found",
        "lyrics": "",
        "source": f"Hugging Face dataset ({DATASET_NAME})",
        "message": message,
    }


def search_lyrics_in_dataset(artist: str, title: str) -> dict[str, str]:
    require_dependency(load_dataset, "datasets")

    artist_name = artist.strip()
    song_title = title.strip()
    if not artist_name or not song_title:
        raise ValueError("Artist name and song title are both required.")

    normalized_artist = normalize_text(artist_name)
    normalized_title = normalize_text(song_title)
    max_rows = get_dataset_search_limit()

    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    partial_match: dict[str, str] | None = None

    for scanned_rows, row in enumerate(dataset, start=1):
        if scanned_rows > max_rows:
            break

        row_title = str(row.get("title", "")).strip()
        row_artist = str(row.get("artist", "")).strip()
        if not row_title or not row_artist:
            continue

        row_title_normalized = normalize_text(row_title)
        row_artist_normalized = normalize_text(row_artist)

        exact_title = row_title_normalized == normalized_title
        exact_artist = row_artist_normalized == normalized_artist
        title_overlap = normalized_title in row_title_normalized or row_title_normalized in normalized_title
        artist_overlap = (
            normalized_artist in row_artist_normalized or row_artist_normalized in normalized_artist
        )

        if exact_title and exact_artist:
            song_data = build_dataset_song_data(row)
            if not song_data["lyrics"]:
                return {
                    **song_data,
                    "message": (
                        f"Dataset match found for '{artist_name} - {song_title}', but the lyrics field was empty. "
                        "Paste lyrics into Manual Lyrics Paste and click Analyze again."
                    ),
                }
            return song_data

        if partial_match is None and ((exact_title and artist_overlap) or (exact_artist and title_overlap)):
            partial_match = build_dataset_song_data(row)

    if partial_match is not None and partial_match["lyrics"]:
        return partial_match

    return build_dataset_not_found_result(artist_name, song_title, max_rows)


def build_manual_song_data(artist: str, title: str, manual_lyrics: str) -> dict[str, str]:
    artist_name = safe_display(artist, "Unknown artist")
    song_title = safe_display(title, "Untitled song")
    lyrics = clean_lyrics(manual_lyrics)

    if not lyrics:
        raise ValueError("Manual lyrics input is empty.")

    return {
        "title": song_title,
        "artist": artist_name,
        "year": "Unknown",
        "tag": "manual",
        "lyrics": lyrics,
        "source": "Manual lyrics paste",
    }


def empty_album_artwork_result() -> dict[str, str]:
    return {
        "artwork_url": "",
        "collection_name": "",
        "track_view_url": "",
    }


def get_album_artwork(artist: str, title: str) -> dict[str, str]:
    if requests is None:
        return empty_album_artwork_result()

    artist_name = artist.strip()
    song_title = title.strip()
    if not artist_name or not song_title:
        return empty_album_artwork_result()

    try:
        response = requests.get(
            ITUNES_SEARCH_URL,
            params={
                "term": f"{artist_name} {song_title}",
                "entity": "song",
                "limit": 1,
            },
            timeout=8,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return empty_album_artwork_result()

    results = payload.get("results", [])
    if not results:
        return empty_album_artwork_result()

    result = results[0]
    artwork_url = str(result.get("artworkUrl100", "")).strip()
    if artwork_url:
        artwork_url = artwork_url.replace("100x100bb", "600x600bb")

    return {
        "artwork_url": artwork_url,
        "collection_name": str(result.get("collectionName", "")).strip(),
        "track_view_url": str(result.get("trackViewUrl", "")).strip(),
    }


def empty_spotify_metadata_result() -> dict[str, Any]:
    return {
        "popularity": None,
        "album": None,
        "duration": None,
    }


def get_spotify_dataset() -> Any:
    require_dependency(load_dataset, "datasets")

    global _spotify_dataset
    if _spotify_dataset is None:
        dataset = load_dataset(SPOTIFY_DATASET_NAME, split="train")
        _spotify_dataset = dataset
    return _spotify_dataset


def format_duration(duration_ms: Any) -> str:
    if safe_display(duration_ms, "") == "":
        return "Unknown"

    if isinstance(duration_ms, str) and ":" in duration_ms:
        return safe_display(duration_ms, "Unknown")

    try:
        total_seconds = int(duration_ms) // 1000
    except (TypeError, ValueError):
        return "Unknown"

    if total_seconds <= 0:
        return "Unknown"

    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"


def format_metadata_value(label: str, value: Any, fallback: str = "Unknown") -> str:
    return f"{label}: {safe_display(value, fallback)}"


def format_popularity(popularity: Any) -> str:
    return safe_display(popularity, "Not available")


def metadata_is_available(value: Any) -> bool:
    return safe_display(value, "") != ""


def metadata_unavailable_note(values: list[Any], threshold: int = 3) -> str:
    missing_count = sum(1 for value in values if not metadata_is_available(value))
    if missing_count >= threshold:
        return '<div class="metadata-note">Metadata unavailable for this track</div>'
    return ""


def get_spotify_metadata(artist: str, title: str) -> dict[str, Any]:
    artist_name = artist.strip().lower()
    song_title = title.strip().lower()
    if not artist_name or not song_title:
        return {
            "popularity": None,
            "album": None,
            "duration": None,
        }

    matches: list[dict[str, Any]] = []

    try:
        dataset = get_spotify_dataset()
        for row in dataset:
            track_name = str(row.get("track_name", "")).strip().lower()
            artists = str(row.get("artists", "")).strip().lower()
            if song_title in track_name and artist_name in artists:
                matches.append(row)
    except Exception:
        return {
            "popularity": None,
            "album": None,
            "duration": None,
        }

    if not matches:
        return {
            "popularity": None,
            "album": None,
            "duration": None,
        }

    best = max(matches, key=lambda row: row.get("popularity", 0))

    return {
        "popularity": best.get("popularity"),
        "album": best.get("album_name"),
        "duration": best.get("duration_ms"),
    }


def interpret_spotify_popularity(popularity: Any) -> str:
    if popularity is None:
        return "Not available"

    try:
        popularity_score = int(popularity)
    except (TypeError, ValueError):
        return "Not available"

    if popularity_score > 70:
        return "high popularity"
    if popularity_score >= 40:
        return "medium popularity"
    return "low popularity"


def score_categories(tokens: list[str], categories: dict[str, set[str]]) -> Counter[str]:
    scores: Counter[str] = Counter()
    for token in tokens:
        for name, lexicon in categories.items():
            if token in lexicon:
                scores[name] += 1
    return scores


def pick_keywords(tokens: list[str], limit: int = 8) -> list[str]:
    filtered = [token for token in tokens if token not in STOPWORDS and len(token) > 2]
    return [word for word, _ in Counter(filtered).most_common(limit)]


def build_interpretation(
    dominant_emotion: str,
    sentiment: str,
    themes: list[str],
) -> str:
    joined_themes = ", ".join(themes[:3]) if themes else "emotional self-expression"
    return (
        f"The lyrics feel primarily {dominant_emotion}, with a {sentiment} tone and recurring focus on "
        f"{joined_themes}. The emotional pattern suggests a song that can be used to understand how imagery, "
        "tone, and repeated motifs shape listener mood."
    )


def build_emotion_justification(dominant_emotion: str, keywords: list[str], sentiment: str) -> str:
    if keywords:
        evidence = ", ".join(keywords[:4])
        return (
            f"Selected {dominant_emotion} because the lyrics emphasize {evidence} "
            f"with an overall {sentiment} tone."
        )
    return f"Selected {dominant_emotion} based on the overall {sentiment} tone and recurring lyric imagery."


def build_emotion_nuance(dominant_emotion: str, sentiment: str) -> str:
    nuance_map = {
        "joy": "bright joy",
        "sadness": "heavy sadness",
        "nostalgia": "bittersweet nostalgia",
        "longing": "hopeful longing" if sentiment == "positive" else "aching longing",
        "anger": "restless anger",
        "confidence": "bold confidence",
        "melancholy": "quiet melancholy",
        "hope": "steady hope",
        "love": "resurgent love" if sentiment == "positive" else "wounded love",
        "tension": "rising tension",
        "desire": "urgent desire",
        "regret": "lingering regret",
        "loneliness": "quiet loneliness",
        "freedom": "open freedom",
        "fear": "shadowed fear",
    }
    return nuance_map.get(dominant_emotion, "quiet melancholy")


def recommend_use_cases(dominant_emotion: str, sentiment: str) -> list[str]:
    recommendations = [
        "playlist mood tagging",
        "songwriting or creative-writing reference",
    ]

    if dominant_emotion in {"sadness", "longing"}:
        recommendations.append("reflective, breakup, or late-night listening curation")
    elif dominant_emotion in {"joy", "confidence"}:
        recommendations.append("celebration, workout, or confidence-boosting playlist curation")
    else:
        recommendations.append("emotion-aware music recommendation experiments")

    return recommendations[:3]


def recommend_similar_songs(dominant_emotion: str, themes: list[str]) -> list[str]:
    theme_text = ", ".join(themes[:2]) if themes else "a similar emotional mood"
    recommendations = {
        "joy": [
            f"Happy — Pharrell Williams: upbeat energy and {theme_text}.",
            f"Good as Hell — Lizzo: confident positivity with a bright emotional lift.",
            f"Walking on Sunshine — Katrina and the Waves: high-valence joy and celebratory momentum.",
        ],
        "sadness": [
            f"Someone Like You — Adele: direct sadness and emotional closure.",
            f"Skinny Love — Bon Iver: fragile heartbreak with subdued intensity.",
            f"The Night We Met — Lord Huron: sorrowful reflection and loss-centered imagery.",
        ],
        "nostalgia": [
            f"Ribs — Lorde: bittersweet nostalgia and memory-driven emotion.",
            f"1979 — The Smashing Pumpkins: reflective youth imagery and warm distance.",
            f"Castle on the Hill — Ed Sheeran: personal memory and coming-of-age themes.",
        ],
        "longing": [
            f"All I Want — Kodaline: aching longing and emotional vulnerability.",
            f"Stay — Rihanna feat. Mikky Ekko: intimate need and unresolved attachment.",
            f"Back to December — Taylor Swift: longing shaped by memory and remorse.",
        ],
        "anger": [
            f"You Oughta Know — Alanis Morissette: sharp anger and emotional release.",
            f"Before He Cheats — Carrie Underwood: revenge-driven tension and hurt.",
            f"Given Up — Linkin Park: explosive frustration and high arousal.",
        ],
        "confidence": [
            f"Stronger — Kanye West: assertive confidence and high-energy drive.",
            f"Run the World (Girls) — Beyonce: empowerment and bold self-possession.",
            f"Unstoppable — Sia: resilience-focused confidence with an anthemic feel.",
        ],
        "melancholy": [
            f"Holocene — Bon Iver: quiet melancholy and spacious reflection.",
            f"Liability — Lorde: intimate self-reflection and low-arousal sadness.",
            f"Motion Picture Soundtrack — Radiohead: delicate melancholy and emotional distance.",
        ],
        "hope": [
            f"Fix You — Coldplay: gentle hope after pain and emotional repair.",
            f"Rise Up — Andra Day: resilience and hopeful perseverance.",
            f"Keep Your Head Up — Ben Howard: warm hope and forward motion.",
        ],
        "love": [
            f"All of Me — John Legend: direct romantic love and emotional openness.",
            f"Lover — Taylor Swift: intimate devotion and warm attachment.",
            f"Adore You — Harry Styles: affectionate love with bright valence.",
        ],
        "tension": [
            f"Take Me to Church — Hozier: emotional tension and dramatic intensity.",
            f"Bad Guy — Billie Eilish: controlled tension with dark playfulness.",
            f"Seven Nation Army — The White Stripes: pulsing tension and forceful restraint.",
        ],
        "desire": [
            f"Earned It — The Weeknd: sensual desire and slow-burn atmosphere.",
            f"Adorn — Miguel: romantic desire with smooth warmth.",
            f"Into You — Ariana Grande: high-arousal desire and pop urgency.",
        ],
        "regret": [
            f"Back to December — Taylor Swift: apology-centered regret and reflection.",
            f"Sorry — Justin Bieber: regret framed as direct reconciliation.",
            f"When I Was Your Man — Bruno Mars: remorse and lost-love reflection.",
        ],
        "loneliness": [
            f"Everybody's Got to Learn Sometime — The Korgis: loneliness and emotional distance.",
            f"Dancing On My Own — Robyn: isolation inside a dance-pop frame.",
            f"Only Love Can Hurt Like This — Paloma Faith: lonely heartbreak and dramatic longing.",
        ],
        "freedom": [
            f"Dog Days Are Over — Florence + The Machine: release, movement, and emotional freedom.",
            f"Born to Run — Bruce Springsteen: escape and freedom-seeking energy.",
            f"Free Fallin' — Tom Petty: open-road freedom and reflective ease.",
        ],
        "fear": [
            f"Bury a Friend — Billie Eilish: fear, darkness, and uneasy atmosphere.",
            f"Disturbia — Rihanna: anxious tension and high-arousal fear imagery.",
            f"Mad World — Gary Jules: quiet fear and alienated reflection.",
        ],
    }
    return recommendations.get(dominant_emotion, recommendations["melancholy"])


def analyze_lyrics_rule_based(lyrics: str) -> dict[str, Any]:
    tokens = tokenize(lyrics)
    token_count = len(tokens) or 1

    emotion_scores = score_categories(tokens, EMOTION_LEXICONS)
    theme_scores = score_categories(tokens, THEME_LEXICONS)

    positive_score = sum(1 for token in tokens if token in POSITIVE_WORDS)
    negative_score = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    high_arousal_score = sum(1 for token in tokens if token in HIGH_AROUSAL_WORDS)
    exclamation_count = lyrics.count("!")
    uppercase_words = len(re.findall(r"\b[A-Z]{3,}\b", lyrics))

    if emotion_scores:
        top_emotion, top_score = emotion_scores.most_common(1)[0]
        dominant_emotion = normalize_emotion_label(top_emotion)
        emotion_confidence = round(
            clamp(0.45 + (top_score / max(sum(emotion_scores.values()), 1)) * 0.45, 0.0, 0.95),
            2,
        )
    elif positive_score > negative_score:
        dominant_emotion = "joy"
        emotion_confidence = 0.55
    elif negative_score > positive_score:
        dominant_emotion = "sadness"
        emotion_confidence = 0.55
    else:
        dominant_emotion = "melancholy"
        emotion_confidence = 0.35

    sentiment_delta = positive_score - negative_score
    if sentiment_delta >= 3:
        sentiment = "positive"
    elif sentiment_delta <= -3:
        sentiment = "negative"
    elif positive_score or negative_score:
        sentiment = "neutral"
    else:
        sentiment = "neutral"

    intensity_score = (sum(emotion_scores.values()) * 6) + (exclamation_count * 5) + (uppercase_words * 4)
    emotional_intensity = int(clamp(20 + (intensity_score / token_count) * 100, 0, 100))
    normalized_valence = (sentiment_delta / max(positive_score + negative_score, 1) + 1) / 2
    valence = round(clamp(normalized_valence, 0.0, 1.0), 2)
    arousal = round(
        clamp((high_arousal_score + exclamation_count + uppercase_words) / max(token_count / 1.5, 1), 0.0, 1.0),
        2,
    )

    key_themes = [theme for theme, _ in theme_scores.most_common(4)] or ["emotional self-expression"]
    keywords = pick_keywords(tokens)
    emotion_justification = build_emotion_justification(dominant_emotion, keywords, sentiment)
    emotion_nuance = build_emotion_nuance(dominant_emotion, sentiment)

    return {
        "analysis_method": "rule-based fallback",
        "dominant_emotion": dominant_emotion,
        "dominant_emotion_category": dominant_emotion,
        "emotion_nuance": emotion_nuance,
        "emotion_confidence": emotion_confidence,
        "emotion_justification": emotion_justification,
        "justification": emotion_justification,
        "sentiment": sentiment,
        "emotional_intensity": emotional_intensity,
        "valence": valence,
        "arousal": arousal,
        "key_themes": key_themes,
        "keywords": keywords,
        "interpretation": build_interpretation(dominant_emotion, sentiment, key_themes),
        "recommendation_use_cases": recommend_use_cases(dominant_emotion, sentiment),
        "similar_recommended_songs": recommend_similar_songs(dominant_emotion, key_themes),
    }


def build_analysis_prompt(song_data: dict[str, str]) -> str:
    return (
        "Analyze the following song lyrics and return valid JSON only with these keys: "
        "dominant_emotion_category, emotion_nuance, emotion_confidence, justification, sentiment, "
        "emotional_intensity, valence, arousal, key_themes, keywords, interpretation, "
        "recommendation_use_cases, similar_recommended_songs.\n"
        "Rules:\n"
        f"- dominant_emotion_category must be exactly one of: {', '.join(ALLOWED_EMOTIONS)}\n"
        "- do not invent emotion labels outside that list\n"
        "- choose the closest matching emotion from the list when multiple emotions are present\n"
        "- emotion_nuance must be a natural 2-4 word phrase, and must not replace the category\n"
        "- emotion_confidence must be a float from 0.0 to 1.0\n"
        "- justification must be 1-2 short sentences explaining why the emotion was selected\n"
        "- sentiment must be exactly one of: positive, negative, neutral\n"
        "- emotional_intensity must be an integer from 0 to 100\n"
        "- valence must be a float from 0.0 to 1.0\n"
        "- arousal must be a float from 0.0 to 1.0\n"
        "- key_themes, keywords, and recommendation_use_cases must be arrays\n"
        "- interpretation must be 2-3 sentences\n"
        "- recommendation_use_cases must contain exactly 3 items\n"
        "- similar_recommended_songs must contain 3-5 strings formatted as "
        "'Song Title — Artist: short reason'\n"
        "- recommend songs based on emotion category, nuance, valence/arousal, themes, and general musical similarity\n"
        "- avoid quoting long parts of the lyrics\n\n"
        f"Artist: {song_data['artist']}\n"
        f"Title: {song_data['title']}\n"
        f"Year: {song_data['year']}\n"
        f"Tag: {song_data['tag']}\n"
        f"Lyrics:\n{song_data['lyrics']}"
    )


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise ValueError("Gemini returned a non-JSON response.")
        return json.loads(match.group(0))


def normalize_gemini_analysis(data: dict[str, Any], model_name: str) -> dict[str, Any]:
    emotional_intensity = data.get("emotional_intensity", data.get("intensity", 0))
    dominant_emotion = normalize_emotion_label(
        data.get("dominant_emotion_category", data.get("dominant_emotion", "melancholy"))
    )
    emotion_nuance = str(data.get("emotion_nuance", "")).strip()
    if not emotion_nuance:
        emotion_nuance = build_emotion_nuance(dominant_emotion, normalize_sentiment_label(data.get("sentiment", "neutral")))
    emotion_confidence = data.get("emotion_confidence", data.get("confidence", 0.0))
    key_themes = data.get("key_themes", data.get("themes", []))
    justification = str(data.get("justification", data.get("emotion_justification", ""))).strip()
    if not justification:
        justification = f"Selected {dominant_emotion} based on the lyric tone and imagery."
    recommended_songs = data.get("similar_recommended_songs", data.get("recommended_songs", []))

    return {
        "analysis_method": f"gemini ({model_name})",
        "dominant_emotion": dominant_emotion,
        "dominant_emotion_category": dominant_emotion,
        "emotion_nuance": emotion_nuance,
        "emotion_confidence": round(clamp(float(emotion_confidence), 0.0, 1.0), 2),
        "emotion_justification": justification,
        "justification": justification,
        "sentiment": normalize_sentiment_label(data.get("sentiment", "neutral")),
        "emotional_intensity": int(clamp(float(emotional_intensity), 0, 100)),
        "valence": round(clamp(float(data.get("valence", 0.0)), 0.0, 1.0), 2),
        "arousal": round(clamp(float(data.get("arousal", 0.0)), 0.0, 1.0), 2),
        "key_themes": [str(item).strip() for item in key_themes if str(item).strip()],
        "keywords": [str(item).strip() for item in data.get("keywords", []) if str(item).strip()],
        "interpretation": str(data.get("interpretation", "")).strip(),
        "recommendation_use_cases": [
            str(item).strip()
            for item in data.get("recommendation_use_cases", [])
            if str(item).strip()
        ],
        "similar_recommended_songs": [
            str(item).strip()
            for item in recommended_songs
            if str(item).strip()
        ],
    }


def analyze_lyrics_with_gemini(song_data: dict[str, str]) -> dict[str, Any]:
    require_dependency(genai, "google-genai")

    api_key = get_env_value("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to the project's .env file to use Gemini.")

    model_name = get_env_value("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=build_analysis_prompt(song_data),
    )

    response_text = getattr(response, "text", "") or ""
    if not response_text.strip():
        raise RuntimeError("Gemini returned an empty response.")

    return normalize_gemini_analysis(extract_json_object(response_text), model_name)


def analyze_song_lyrics(song_data: dict[str, str]) -> dict[str, Any]:
    if get_env_value("GEMINI_API_KEY"):
        try:
            return analyze_lyrics_with_gemini(song_data)
        except Exception as exc:
            fallback = analyze_lyrics_rule_based(song_data["lyrics"])
            fallback["analysis_method"] = f"rule-based fallback (Gemini unavailable: {exc})"
            return fallback
    return analyze_lyrics_rule_based(song_data["lyrics"])


def format_analysis(song_data: dict[str, str], analysis: dict[str, Any]) -> str:
    themes_text = ", ".join(analysis["key_themes"]) or "n/a"
    keywords_text = ", ".join(analysis["keywords"]) or "n/a"
    mood_cluster = assign_mood_cluster(analysis.get("valence"), analysis.get("arousal"))
    use_cases = list(analysis["recommendation_use_cases"][:3])
    default_use_cases = [
        "playlist mood tagging",
        "songwriting or creative-writing reference",
        "emotion-aware music recommendation experiments",
    ]
    use_cases.extend(default_use_cases[len(use_cases):])
    use_cases_text = "\n".join(f"- {item}" for item in use_cases[:3])
    similar_songs = list(analysis.get("similar_recommended_songs", [])[:5])
    if len(similar_songs) < 3:
        similar_songs.extend(recommend_similar_songs(analysis["dominant_emotion"], analysis["key_themes"])[len(similar_songs):])
    similar_songs_text = "\n".join(f"- {item}" for item in similar_songs[:5])
    media_block = ""
    if song_data.get("artwork_url"):
        media_block += f"![Album cover]({song_data['artwork_url']})\n\n"
    if song_data.get("collection_name"):
        media_block += f"Album / Collection: {song_data['collection_name']}  \n"
    if song_data.get("track_view_url"):
        media_block += f"iTunes Track: [Open in iTunes]({song_data['track_view_url']})  \n"
    if media_block:
        media_block += "\n"

    return f"""{media_block}Dominant Emotion Category: {analysis['dominant_emotion_category']}  
Emotion Nuance: {analysis.get('emotion_nuance', 'quiet melancholy')}  
Emotion Confidence: {analysis.get('emotion_confidence', 0.0)}  
Justification: {analysis.get('justification', analysis.get('emotion_justification', 'n/a'))}  

Sentiment: {analysis['sentiment']}  
Emotional Intensity: {analysis['emotional_intensity']}  
Valence: {analysis['valence']}  
Arousal: {analysis['arousal']}  

Mood Cluster:
- Cluster Label: {mood_cluster['cluster_label']}
- Explanation: {mood_cluster['explanation']}

Key Themes: {themes_text}  
Keywords: {keywords_text}  

Interpretation: {analysis['interpretation']}  

Recommendation Use Cases:
{use_cases_text}

Similar / Recommended Songs:
{similar_songs_text}
"""


def get_display_similar_songs(analysis: dict[str, Any]) -> list[str]:
    similar_songs = list(analysis.get("similar_recommended_songs", [])[:5])
    if len(similar_songs) < 3:
        fallback_songs = recommend_similar_songs(
            analysis.get("dominant_emotion", "melancholy"),
            analysis.get("key_themes", []),
        )
        similar_songs.extend(fallback_songs[len(similar_songs):])
    return similar_songs[:5]


def format_similar_songs_markdown(similar_songs: list[str]) -> str:
    if not similar_songs:
        return "### 🎵 Similar Songs\nRun an analysis to generate recommendations."
    items = "\n".join(f"{index}. {song}" for index, song in enumerate(similar_songs, start=1))
    return f"### 🎵 Similar Songs\n{items}"


def build_album_cover_html(song_data: dict[str, Any]) -> str:
    title = escape(safe_display(song_data.get("title"), "Analyzed song"), quote=True)
    artwork_url = str(song_data.get("artwork_url", "")).strip()
    if artwork_url:
        safe_artwork_url = escape(artwork_url, quote=True)
        return f'<div class="cover-frame"><img src="{safe_artwork_url}" alt="{title} album cover"></div>'
    return '<div class="cover-frame"><div class="cover-placeholder">🎵</div></div>'


def build_track_card_html(
    song_data: dict[str, Any] | None = None,
    analysis: dict[str, Any] | None = None,
    mood_cluster: dict[str, str] | None = None,
) -> str:
    if not song_data or not analysis or not mood_cluster:
        return """
        <div class="track-card">
            <div class="cover-frame"><div class="cover-placeholder">🎧</div></div>
            <div>
                <div class="track-kicker">Ready when you are</div>
                <div class="track-title">Search a song</div>
                <div class="track-artist">Enter artist and title, or paste lyrics manually.</div>
                <div class="pill-row">
                    <span class="pill accent">💚 Gemini emotion analysis</span>
                    <span class="pill purple">📍 Valence-arousal map</span>
                    <span class="pill">🎵 Similarity recommendations</span>
                </div>
            </div>
        </div>
        """

    title_text = safe_display(song_data.get("title"), "Untitled song")
    artist_text = safe_display(song_data.get("artist"), "Unknown artist")
    title = escape(title_text)
    artist = escape(artist_text)
    emotion = escape(str(analysis.get("dominant_emotion_category", "melancholy")).title())
    confidence = clamp(float(analysis.get("emotion_confidence", 0.0)), 0.0, 1.0)
    confidence_percent = int(round(confidence * 100))
    mood_label = escape(str(mood_cluster.get("cluster_label", "mixed or reflective")).title())
    source = safe_display(song_data.get("source"), "Runtime analysis")
    album = safe_display(
        song_data.get("collection_name") or song_data.get("album"),
        "Not available",
    )
    duration = format_duration(song_data.get("duration"))
    popularity = format_popularity(song_data.get("popularity"))
    year = safe_display(song_data.get("year"), "Unknown")
    metadata_note = metadata_unavailable_note(
        [
            song_data.get("collection_name") or song_data.get("album"),
            song_data.get("duration"),
            song_data.get("popularity"),
            song_data.get("artist"),
            song_data.get("year"),
        ],
        threshold=3,
    )
    track_link = str(song_data.get("track_view_url", "")).strip()
    link_html = ""
    if track_link:
        safe_track_link = escape(track_link, quote=True)
        link_html = f'<span><a href="{safe_track_link}" target="_blank" rel="noopener noreferrer">Open in iTunes</a></span>'

    return f"""
    <div class="track-card">
        {build_album_cover_html(song_data)}
        <div>
            <div class="track-kicker">Analyzed Track</div>
            <div class="track-title">{title}</div>
            <div class="track-artist">{artist}</div>
            <div class="pill-row">
                <span class="pill accent">💚 {emotion}</span>
                <span class="pill purple">🔥 {confidence_percent}% confidence</span>
                <span class="pill">📍 {mood_label}</span>
            </div>
            <div class="track-meta">
                <span>{escape(format_metadata_value("Album", album, "Not available"))}</span>
                <span>{escape(format_metadata_value("Duration", duration))}</span>
                <span>{escape(format_metadata_value("Popularity", popularity, "Not available"))}</span>
                <span>{escape(format_metadata_value("Artist", artist_text))}</span>
                <span>{escape(format_metadata_value("Year", year))}</span>
                <span>{escape(format_metadata_value("Source", source))}</span>
                {link_html}
            </div>
            {metadata_note}
        </div>
    </div>
    """


def run_full_analysis(artist: str, title: str, manual_lyrics: str = "") -> dict[str, Any]:
    if manual_lyrics.strip():
        song_data = build_manual_song_data(artist, title, manual_lyrics)
    else:
        song_data = search_lyrics_in_dataset(artist=artist, title=title)
        if not song_data.get("lyrics"):
            raise LookupError(song_data.get("message", "Song lyrics were not found."))

    song_data.update(get_album_artwork(song_data["artist"], song_data["title"]))
    song_data.update(get_spotify_metadata(song_data["artist"], song_data["title"]))

    analysis = analyze_song_lyrics(song_data)
    similar_songs = get_similar_songs_faiss(song_data["lyrics"], top_k=5)
    if similar_songs:
        analysis["similar_recommended_songs"] = format_embedding_recommendations(similar_songs)

    mood_cluster = assign_mood_cluster(analysis.get("valence"), analysis.get("arousal"))
    try:
        plot_path = create_valence_arousal_plot(
            analysis.get("valence"),
            analysis.get("arousal"),
            song_data["title"],
        )
    except Exception:
        plot_path = None

    return {
        "song_data": song_data,
        "analysis": analysis,
        "mood_cluster": mood_cluster,
        "plot_path": plot_path,
        "formatted_analysis": format_analysis(song_data, analysis),
        "similar_songs": get_display_similar_songs(analysis),
    }


def analyze_request(artist: str, title: str, manual_lyrics: str = "") -> tuple[str, str | None]:
    try:
        result = run_full_analysis(artist, title, manual_lyrics)
        return result["formatted_analysis"], result["plot_path"]
    except LookupError as exc:
        return f"## Song Not Found\n\n{exc}", None
    except Exception as exc:
        return f"## Error\n\n{exc}", None


def empty_ui_values(status_message: str) -> tuple[str, str, float, float, str | None, str, str]:
    return (
        status_message,
        build_track_card_html(),
        0.0,
        0.0,
        None,
        "### 🎵 Similar Songs\nRun an analysis to generate recommendations.",
        "",
    )


def analyze_request_ui(
    artist: str,
    title: str,
    manual_lyrics: str = "",
) -> tuple[str, str, float, float, str | None, str, str]:
    try:
        result = run_full_analysis(artist, title, manual_lyrics)
    except LookupError as exc:
        return empty_ui_values(f"### ⚠️ Song Not Found\n{exc}")
    except Exception as exc:
        return empty_ui_values(f"### ⚠️ Error\n{exc}")

    song_data = result["song_data"]
    analysis = result["analysis"]
    mood_cluster = result["mood_cluster"]
    valence = round(clamp(float(analysis.get("valence", 0.0)), 0.0, 1.0), 2)
    arousal = round(clamp(float(analysis.get("arousal", 0.0)), 0.0, 1.0), 2)
    status = (
        f"✅ Analysis complete · {analysis.get('analysis_method', 'runtime analysis')} · "
        f"{mood_cluster['cluster_label']}"
    )
    return (
        status,
        build_track_card_html(song_data, analysis, mood_cluster),
        valence,
        arousal,
        result["plot_path"],
        format_similar_songs_markdown(result["similar_songs"]),
        result["formatted_analysis"],
    )


def set_analysis_loading() -> tuple[Any, str]:
    return (
        gr.update(value="Analyzing...", interactive=False),
        "⏳ Searching lyrics, analyzing emotion, and building the map...",
    )


def reset_analysis_button() -> Any:
    return gr.update(value="Analyze Track", interactive=True)


def gradio_uses_launch_css() -> bool:
    try:
        major_version = int(str(gr.__version__).split(".", maxsplit=1)[0])
    except (AttributeError, TypeError, ValueError):
        return False
    return major_version >= 6


def build_app() -> Any:
    require_dependency(gr, "gradio")
    blocks_kwargs = {"title": APP_TITLE}
    if not gradio_uses_launch_css():
        blocks_kwargs["css"] = SPOTIFY_CSS

    with gr.Blocks(**blocks_kwargs) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                """
                <section class="hero">
                    <h1>🎧 Emotion-Aware Music Analysis System</h1>
                    <p>
                        Search lyrics, run Gemini-powered emotion analysis, map the track into
                        valence-arousal space, and discover similar songs in a polished music-app workflow.
                    </p>
                </section>
                """
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=4, elem_classes=["input-card"]):
                    gr.Markdown("## 🎵 Track Input", elem_classes=["section-title"])
                    artist_input = gr.Textbox(
                        label="Artist name",
                        placeholder="Adele",
                    )
                    title_input = gr.Textbox(
                        label="Song title",
                        placeholder="Hello",
                    )
                    gr.Markdown("**OR paste lyrics manually**")
                    manual_lyrics_input = gr.Textbox(
                        label="Manual lyrics input",
                        lines=9,
                        placeholder="Paste lyrics here if the dataset search does not find the song.",
                    )
                    analyze_button = gr.Button(
                        "Analyze Track",
                        variant="primary",
                        elem_classes=["analyze-button"],
                    )
                    status_output = gr.Markdown(
                        "Ready. Search a song or paste lyrics to begin.",
                        elem_classes=["status-text"],
                    )

                with gr.Column(scale=6, elem_classes=["track-card-shell"]):
                    gr.Markdown("## Now Analyzing", elem_classes=["section-title"])
                    track_card_output = gr.HTML(build_track_card_html())

            with gr.Row(equal_height=False):
                with gr.Column(scale=4, elem_classes=["metric-card"]):
                    gr.Markdown("## 💚 Emotion Metrics", elem_classes=["section-title"])
                    valence_output = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.01,
                        label="Valence (0 = negative, 1 = positive)",
                        interactive=False,
                        elem_classes=["slider-wrap"],
                    )
                    arousal_output = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.01,
                        label="Arousal (0 = calm, 1 = energetic)",
                        interactive=False,
                        elem_classes=["slider-wrap"],
                    )

                with gr.Column(scale=6, elem_classes=["plot-card"]):
                    gr.Markdown("## 📍 Emotion Space", elem_classes=["section-title"])
                    plot_output = gr.Image(
                        label="Valence / Arousal Map",
                        type="filepath",
                        height=420,
                    )

            with gr.Row(equal_height=False):
                with gr.Column(scale=5, elem_classes=["similar-card"]):
                    similar_songs_output = gr.Markdown(
                        "### 🎵 Similar Songs\nRun an analysis to generate recommendations."
                    )

                with gr.Column(scale=7, elem_classes=["details-card"]):
                    with gr.Accordion("Detailed Structured Output", open=False):
                        detailed_output = gr.Markdown()

            analysis_flow = analyze_button.click(
                fn=set_analysis_loading,
                outputs=[analyze_button, status_output],
                queue=False,
            )
            analysis_flow = analysis_flow.then(
                fn=analyze_request_ui,
                inputs=[artist_input, title_input, manual_lyrics_input],
                outputs=[
                    status_output,
                    track_card_output,
                    valence_output,
                    arousal_output,
                    plot_output,
                    similar_songs_output,
                    detailed_output,
                ],
            )
            analysis_flow.then(
                fn=reset_analysis_button,
                outputs=analyze_button,
                queue=False,
            )

    return demo


if __name__ == "__main__":
    queued_demo = build_app().queue()
    if gradio_uses_launch_css():
        queued_demo.launch(css=SPOTIFY_CSS)
    else:
        queued_demo.launch()
