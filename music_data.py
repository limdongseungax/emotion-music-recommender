"""감정별 추천 음악 매핑 데이터 및 이모지 정의."""

# 감정별 이모지 매핑
EMOTION_EMOJIS: dict[str, str] = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲",
}

# 감정별 한국어 명칭 (UI 표시용 보조 딕셔너리)
EMOTION_LABELS_KO: dict[str, str] = {
    "angry": "분노 (Angry)",
    "disgust": "불쾌 (Disgust)",
    "fear": "두려움/불안 (Fear)",
    "happy": "행복 (Happy)",
    "neutral": "평온 (Neutral)",
    "sad": "슬픔 (Sad)",
    "surprise": "놀람 (Surprise)",
}

# 감정별 추천 음악 데이터 (각 5곡: 한국 노래 + 팝송 혼합)
EMOTION_MUSIC: dict[str, list[dict[str, str]]] = {
    "angry": [
        {"title": "TOMBOY", "artist": "(여자)아이들"},
        {"title": "불타오르네 (FIRE)", "artist": "방탄소년단 (BTS)"},
        {"title": "Believer", "artist": "Imagine Dragons"},
        {"title": "Smells Like Teen Spirit", "artist": "Nirvana"},
        {"title": "Numb", "artist": "Linkin Park"},
    ],
    "disgust": [
        {"title": "상쾌해", "artist": "로꼬 (Loco)"},
        {"title": "Attention", "artist": "NewJeans"},
        {"title": "Shake It Off", "artist": "Taylor Swift"},
        {"title": "Good as Hell", "artist": "Lizzo"},
        {"title": "Juice", "artist": "Lizzo"},
    ],
    "fear": [
        {"title": "한숨", "artist": "이하이"},
        {"title": "걱정말아요 그대", "artist": "이적"},
        {"title": "Fix You", "artist": "Coldplay"},
        {"title": "Count on Me", "artist": "Bruno Mars"},
        {"title": "Brave", "artist": "Sara Bareilles"},
    ],
    "happy": [
        {"title": "한 페이지가 될 수 있게", "artist": "DAY6 (데이식스)"},
        {"title": "파이팅 해야지 (Feat. 이영지)", "artist": "부석순 (SEVENTEEN)"},
        {"title": "Happy", "artist": "Pharrell Williams"},
        {"title": "Uptown Funk", "artist": "Mark Ronson (Feat. Bruno Mars)"},
        {"title": "Can't Stop the Feeling!", "artist": "Justin Timberlake"},
    ],
    "neutral": [
        {"title": "Square (2017)", "artist": "백예린 (Yerin Baek)"},
        {"title": "Instagram", "artist": "DEAN (딘)"},
        {"title": "Sunday Morning", "artist": "Maroon 5"},
        {"title": "Paris in the Rain", "artist": "Lauv"},
        {"title": "Best Part (Feat. H.E.R.)", "artist": "Daniel Caesar"},
    ],
    "sad": [
        {"title": "밤편지", "artist": "아이유 (IU)"},
        {"title": "모든 날, 모든 순간", "artist": "폴킴 (Paul Kim)"},
        {"title": "야생화", "artist": "박효신"},
        {"title": "Someone Like You", "artist": "Adele"},
        {"title": "When I Was Your Man", "artist": "Bruno Mars"},
    ],
    "surprise": [
        {"title": "Next Level", "artist": "aespa (에스파)"},
        {"title": "Super Shy", "artist": "NewJeans"},
        {"title": "Blinding Lights", "artist": "The Weeknd"},
        {"title": "Levitating", "artist": "Dua Lipa"},
        {"title": "Starboy", "artist": "The Weeknd"},
    ],
}


def get_music_by_emotion(emotion: str) -> list[dict[str, str]]:
    """감정 문자열을 입력받아 해당 감정의 추천 곡 리스트를 반환합니다."""
    return EMOTION_MUSIC.get(emotion.lower(), EMOTION_MUSIC["neutral"])


def get_emoji_by_emotion(emotion: str) -> str:
    """감정 문자열을 입력받아 해당 감정의 이모지를 반환합니다."""
    return EMOTION_EMOJIS.get(emotion.lower(), "🎵")
