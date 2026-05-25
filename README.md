# 🎧 Emotion-Aware Music Analysis System

> 🇰🇷 *한국어 버전은 아래에서 확인하실 수 있습니다. (Please scroll down for the Korean version)*

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

---

# 🇰🇷 (Korean Version)

# 🎧 감정 인지형 음악 분석 시스템 (Emotion-Aware Music Analysis System)

대규모 언어 모델(LLM)을 활용하여 가사의 감정을 분석하고, 이를 구조화된 감정 공간(Valence-Arousal)에 매핑(Mapping)하는 인공지능 시스템입니다. 지능형 음악 이해, 해석 및 추천 기능을 지원합니다.

---

### 담당 역할 (Role)
- AI 시스템 아키텍처 설계
- LLM 파이프라인 개발
- 감정 모델링 및 보정(Calibration) 레이어 설계
- FAISS 기반 추천 시스템 개발
- 백엔드 연동 및 시스템 배포 (Deployment)

### 기여도 (Contribution)
100% (개인 프로젝트)

---

## 🚀 라이브 데모 (Live Demo)

👉 Hugging Face Space:  
https://huggingface.co/spaces/ALEXJK0901/emotion-aware-music-analysis-system

---

## 🚀 주요 특징 (Key Highlights)

* **Gemini**를 활용한 LLM 기반 감정 분석
* 하이브리드 감정 모델링 (**의미적 감정 + 수치적 공간 + 일관성 레이어**)
* 심리학적 모델 기반 감정 → Valence/Arousal 매핑
* **감정적 일관성(Affective consistency)** 확보를 위한 후처리 보정 레이어 구축
* FAISS 벡터 검색을 활용한 유사 곡 추천
* 감정 공간 시각화 (Valence-Arousal Map)
* 다중 데이터 소스 통합 (가사 + Spotify 형태의 메타데이터 + iTunes API)

---

## 🧠 시스템 아키텍처 (System Architecture)

### 1. 가사 데이터 검색 (Lyrics Retrieval)
* Hugging Face 데이터셋 스트리밍 처리 (`theelderemo/genius-lyrics-cleaned`)
* 전체 데이터셋을 다운로드할 필요 없이 실시간 탐색 가능

### 2. 감정 분석 (Emotion Analysis - LLM)
* Gemini API 활용
* 구조화된 출력 (Structured output):
  * 감정 카테고리 (Emotion category)
  * 감정의 뉘앙스 (Emotion nuance)
  * 신뢰도 점수 (Confidence score)
  * 원시 Valence & Arousal 수치

### 3. 감정 보정 레이어 (Affective Calibration Layer - 핵심 기여도)

감정 카테고리와 수치적 표현 간의 **구조적 일관성(Structural consistency)을 보장**하는 경량화된 후처리 모듈입니다.

포함 내용:
* 감정 → Valence/Arousal 매핑
* 모델 출력값과 매핑값의 적응형 혼합 (Adaptive blending)
* 저각성(Low-arousal) 감정 제약 조건 적용 (예: 그리움, 향수, 우울함)
* 엣지 케이스 처리를 위한 클러스터 재정의 룰 적용

💡 이를 통해 **"그리움(longing)"이 "긴장됨/화남(tense/angry)"으로 분류되는 등의 감정적 불일치를 방지**합니다.

### 4. 감정 모델링 (Emotion Modeling)

하이브리드 파이프라인:
* LLM → 의미론적 감정 이해 (Semantic emotion understanding)
* Mapping → 수치적 변환 (Numeric representation)
* Calibration → 일관성 보정 (Consistency correction)
* Rules → 안정성 및 해석 가능성 확보

### 5. 무드 클러스터링 (Mood Clustering)

Valence-Arousal 공간을 다음 카테고리로 매핑합니다:
* 슬픔 / 차분함 (sad / calm)  
* 활기참 / 즐거움 (energetic / joyful)  
* 평화로움 / 따뜻함 (peaceful / warm)  
* 긴장됨 / 분노 (tense / angry)  
* 복합적 / 회고적 (mixed / reflective)  

### 6. 추천 엔진 (Recommendation Engine)

* Sentence Transformers 기반 임베딩
* FAISS 벡터 탐색 (Vector search)
* 코사인 유사도(Cosine similarity) 폴백 메커니즘

### 7. 시각화 (Visualization)

* Valence vs Arousal 감정 공간 플로팅 (Plotting)
* 분석된 곡의 실시간 위치 시각화 표시

---

## 🎯 활용 분야 (Use Cases)

* 감정 인지형 음악 추천 시스템
* 엔터테인먼트 플랫폼을 위한 AI 기반 콘텐츠 분석
* 감정적 맥락(Emotional context) 기반 플레이리스트 생성
* 음악 제작을 위한 AI 창작 지원 도구
* 미디어 플랫폼을 위한 의사결정 지원 시스템 (Decision-support systems)

---

## 🛠 기술 스택 (Tech Stack)

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

## ⚙️ 주요 기능 (Features)

* `streaming=True` 옵션을 활용한 가사 데이터셋 스트림 검색
* 대소문자 구분 없는 아티스트/제목 매칭 지원
* 수동 가사 입력 폴백(Fallback) 기능
* 구조화된 LLM 기반 감정 분석
* 감정 신뢰도 점수 및 분석 근거 제공
* 통제된 감정 분류 체계(Taxonomy) 적용
* 감정적 일관성을 위한 보정(Calibration) 레이어 처리
* LLM 응답 실패 시 룰 기반 폴백 지원
* Spotify 형태의 메타데이터 강화 (Enrichment)
* iTunes 앨범 아트워크 연동
* FAISS 기반 유사 곡 추천 기능
* 감정 공간(Emotion-space) 시각화

---

## 📂 프로젝트 구조 (Project Structure)

```text
lyrics-emotion-analysis-assistant/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── screenshots/
└── vectorstore/
```

---

## 🔧 설치 및 설정 (Setup)

```bash
cd lyrics-emotion-analysis-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`.env` 파일 생성:

```text
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
HF_DATASET_SEARCH_LIMIT=50000
```

---

## ▶️ 실행 방법 (Run the App)

```bash
python3 app.py
```

터미널에 표시된 로컬 Gradio URL을 열어 확인합니다.

---

## 🔍 작동 원리 (How It Works)

* Hugging Face 데이터셋에서 데이터를 스트리밍합니다 (전체 다운로드 불필요).
* 아티스트와 제목을 기반으로 곡을 매칭합니다.
* 외부 소스를 통해 메타데이터를 강화합니다:
  * iTunes API (아트워크, 앨범 정보)
  * Spotify 데이터셋 (인기도, 곡 길이)
* 실시간 LLM 기반 분석을 수행합니다.
* 결과값의 일관성을 위해 보정(Calibration) 레이어를 거칩니다.
* 구조화된 감정적 인사이트를 생성합니다.
* 임베딩을 통해 유사한 곡을 탐색합니다.
* Valence-Arousal 공간에 감정을 시각화합니다.

---

## 🎵 Spotify 데이터셋 통합 (Spotify Dataset Integration)

* 사용 데이터셋: `maharshipandya/spotify-tracks-dataset`
* 별도의 Spotify API 연동 불필요
* 곡 제목 + 아티스트명을 통한 매칭
* 출력 데이터:
  * 인기도 (popularity)
  * 앨범 정보 (album)
  * 곡 길이 (duration)
  * 해석 (interpretation)

---

## 🖼 앨범 아트워크 연동 (선택 기능)

* 인증이 필요 없는 iTunes Search API 활용
* 앨범 아트워크 및 관련 메타데이터 호출
* 사용자가 자유롭게 켜고 끌 수 있는 선택적 통합 기능

---

## ⚠️ 설계 결정: Live Genius 크롤링 제외 (Design Decision)

기존 Genius 크롤링 방식을 제외한 사유:
* Cloudflare 접속 차단 문제
* 시스템 불안정성

대체 방안:
* 공개 데이터셋 활용 → 시스템의 **재현성(Reproducibility)** 및 **안정성(Stability)** 확보

---

## 🔒 런타임 가사 데이터 처리 (Runtime-Only Lyrics Usage)

* 가사 데이터는 메모리 내에서만 처리됩니다.
* 별도의 DB 저장소나 영구 저장을 수행하지 않습니다.
* 사용자가 수동으로 입력한 가사 데이터는 저장되지 않습니다.
* 본 시스템은 연구 및 교육 목적으로 설계되었습니다.

---

## 📊 향후 개선 과제 (Future Improvements)

* 확률적 감정 매핑 (Soft consistency) 도입
* 전체 데이터셋 인덱싱 구축
* 다국어 지원 (Multilingual support)
* 고도화된 클러스터링 알고리즘 적용
* 실시간 음원 API(Spotify, Apple Music 등) 직접 연동
