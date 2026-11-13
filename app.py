"""NutriScan: detect Indian dishes in a photo and estimate the meal's calories.

    streamlit run app.py
"""
import json
from pathlib import Path

import PIL.Image
import streamlit as st
from ultralytics import YOLO

from nutrition import display_name, summarize

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "nutriscan-yolo11s.pt"
CARD_PATH = ROOT / "models" / "model_card.json"

st.set_page_config(page_title="NutriScan", page_icon="🍛", layout="wide")
st.title("🍛 NutriScan: Indian food detection")
st.caption("Upload a photo of a thali or plate. A YOLO11 model finds each dish and estimates the calories.")


@st.cache_resource
def load_model() -> YOLO:
    return YOLO(str(MODEL_PATH))


if not MODEL_PATH.exists():
    st.error(f"Model weights not found at {MODEL_PATH.relative_to(ROOT)}.")
    st.stop()

model = load_model()

with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Food photo", type=["jpg", "jpeg", "png", "webp"])
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                     help="Lower finds more dishes but adds false positives.")
    iou = st.slider("Overlap filter (NMS IoU)", 0.1, 0.9, 0.45, 0.05,
                    help="Lower merges overlapping boxes, e.g. idlis touching each other.")
    with st.expander(f"Detectable dishes ({len(model.names)})"):
        st.write(", ".join(display_name(n) for n in model.names.values()))
    if CARD_PATH.exists():
        card = json.loads(CARD_PATH.read_text())
        st.caption(f"Model: {card['architecture']} · validation mAP@0.5 = {card['val']['mAP50']:.2f}")

if not uploaded:
    st.info("Upload a photo in the sidebar to start.")
    st.stop()

img = PIL.Image.open(uploaded).convert("RGB")
result = model.predict(source=img, conf=conf, iou=iou, verbose=False)[0]
labels = [model.names[int(c)] for c in result.boxes.cls]
meal = summarize(labels)

left, right = st.columns([2, 1])
with left:
    st.image(result.plot()[..., ::-1], caption=f"{len(labels)} item(s) detected", use_container_width=True)  # BGR -> RGB

with right:
    st.subheader("Meal summary")
    if not meal.lines:
        st.warning("No dishes detected. Try a lower confidence threshold or a closer photo.")
    else:
        st.table([{"Dish": ln.name, "Qty": ln.count, "kcal / serving": ln.kcal_each, "kcal": ln.kcal} for ln in meal.lines])
        st.metric("Estimated total", f"{meal.total_kcal} kcal")
        if meal.is_high_calorie:
            st.warning("High-calorie meal. Consider smaller portions.")
        else:
            st.success("Within a typical single-meal range.")
        st.caption("Estimates use one standard serving per detected item; actual portions vary.")

with st.expander("Per-detection confidence"):
    st.table([{"Dish": display_name(model.names[int(c)]), "Confidence": f"{float(p):.2f}"}
              for c, p in zip(result.boxes.cls, result.boxes.conf)] or [{"Dish": "-", "Confidence": "-"}])
