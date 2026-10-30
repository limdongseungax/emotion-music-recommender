from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from music_data import (
    EMOTION_EMOJIS,
    EMOTION_LABELS_KO,
    EMOTION_MUSIC,
    get_emoji_by_emotion,
    get_music_by_emotion,
)

# 1. 페이지 기본 설정 (wide 레이아웃)
st.set_page_config(
    page_title="🎵 표정으로 음악 추천받기",
    page_icon="🎵",
    layout="wide",
)

# 상수 설정
MODEL_PATH = Path("models/best_model.pth")
DEVICE = torch.device("cpu")
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMAGE_SIZE = 96
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@st.cache_resource
def load_emotion_model(model_path: Path = MODEL_PATH) -> nn.Module:
    """학습된 ResNet18(7클래스) 모델을 CPU로 로드합니다."""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(EMOTION_LABELS))

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """train.py와 동일한 전처리를 수행합니다 (흑백->RGB 3채널, 96x96 리사이즈, ImageNet 정규화)."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(image).unsqueeze(0)  # (1, 3, 96, 96)
    return tensor.to(DEVICE)


@torch.no_grad()
def predict_emotion(model: nn.Module, image: Image.Image) -> tuple[str, float, dict[str, float]]:
    """이미지로부터 가장 확률이 높은 감정, 확률(%), 전체 감정별 확률 딕셔너리를 반환합니다."""
    input_tensor = preprocess_image(image)
    outputs = model(input_tensor)
    probs = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy() * 100.0

    prob_dict = {label: float(prob) for label, prob in zip(EMOTION_LABELS, probs)}
    best_idx = int(probs.argmax())
    best_emotion = EMOTION_LABELS[best_idx]
    best_prob = float(probs[best_idx])

    return best_emotion, best_prob, prob_dict


def render_sidebar() -> None:
    """사이드바에 서비스 설명을 렌더링합니다."""
    with st.sidebar:
        st.header("ℹ️ 서비스 안내")
        st.markdown(
            """
            **🎵 표정으로 음악 추천받기**는 딥러닝(ResNet18) 기반의 얼굴 표정 인식 모델을 활용하여,
            현재 표정에 어울리는 맞춤형 음악을 추천해 드리는 서비스입니다.

            ---
            ### 💡 이용 방법
            1. 얼굴이 나온 사진(`JPG`, `PNG`)을 업로드합니다.
            2. AI 모델이 사진 속 표정을 **7가지 감정** 중 하나로 분석합니다.
            3. 분석된 감정에 딱 맞는 **추천 음악 5곡**을 확인해 보세요!

            ---
            ### 😊 분석 가능한 7가지 감정
            """
        )
        for emotion in EMOTION_LABELS:
            emoji = EMOTION_EMOJIS.get(emotion, "🎵")
            ko_label = EMOTION_LABELS_KO.get(emotion, emotion)
            st.markdown(f"- {emoji} **{ko_label}**")

        st.markdown("---")
        st.caption("Model: ResNet18 (Pretrained, 96×96) | Device: CPU")


def main() -> None:
    render_sidebar()

    # 페이지 제목
    st.title("🎵 표정으로 음악 추천받기")
    st.write("사진을 업로드하면 AI가 표정을 분석해 지금 기분에 어울리는 노래 5곡을 추천해 드립니다.")

    # 모델 파일 존재 여부 확인
    if not MODEL_PATH.exists():
        st.warning(
            f"⚠️ 학습된 모델 파일(`{MODEL_PATH}`)을 찾을 수 없습니다. "
            "먼저 `python train.py`를 실행하여 모델을 학습시켜 주세요."
        )

    # 파일 업로더
    uploaded_file = st.file_uploader(
        "얼굴 표정이 담긴 이미지를 업로드하세요 (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1.3], gap="large")

        with col1:
            st.subheader("📷 업로드된 이미지")
            st.image(image, caption="업로드한 이미지", use_container_width=True)

        with col2:
            st.subheader("🔍 표정 분석 결과")

            if not MODEL_PATH.exists():
                st.error("모델 파일이 없어 예측을 수행할 수 없습니다.")
                return

            with st.spinner("표정을 분석하고 있습니다..."):
                model = load_emotion_model(MODEL_PATH)
                best_emotion, best_prob, prob_dict = predict_emotion(model, image)

            emoji = get_emoji_by_emotion(best_emotion)
            ko_label = EMOTION_LABELS_KO.get(best_emotion, best_emotion)

            # 예측 결과: 감정 이모지 + 감정명 + 확률(%)
            st.success(f"### {emoji} {ko_label} — **{best_prob:.2f}%**")

            # 전체 7개 감정의 확률 막대 그래프
            st.markdown("#### 📊 감정별 예측 확률 (%)")
            chart_df = pd.DataFrame({
                "감정 (Emotion)": [
                    f"{EMOTION_EMOJIS[e]} {e}" for e in EMOTION_LABELS
                ],
                "확률 (%)": [round(prob_dict[e], 2) for e in EMOTION_LABELS],
            }).set_index("감정 (Emotion)")

            st.bar_chart(chart_df, y="확률 (%)", height=300)

        st.divider()

        # 해당 감정의 추천 음악 5곡 표시
        emoji = get_emoji_by_emotion(best_emotion)
        ko_label = EMOTION_LABELS_KO.get(best_emotion, best_emotion)
        st.subheader(f"🎧 {emoji} '{ko_label}' 감정을 위한 추천 플레이리스트 (5곡)")

        recommended_songs = get_music_by_emotion(best_emotion)
        song_cols = st.columns(5)

        for idx, (col, song) in enumerate(zip(song_cols, recommended_songs), start=1):
            with col:
                st.markdown(
                    f"""
                    <div style="padding: 16px; border-radius: 12px; border: 1px solid #e0e0e0; background-color: #f9f9fb; height: 140px;">
                        <div style="font-size: 0.85rem; color: #666; margin-bottom: 6px;">Track {idx}</div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #1f1f1f; margin-bottom: 6px;">🎵 {song['title']}</div>
                        <div style="font-size: 0.9rem; color: #555;">🎤 {song['artist']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
