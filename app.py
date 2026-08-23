"""Local EgoMemory demo UI.

Run with: .\.venv\Scripts\python.exe -m streamlit run app.py
"""

import json
from pathlib import Path

import streamlit as st
import torch
from transformers import CLIPModel, CLIPProcessor

from egomemory.retrieval.multimodal import MultimodalRetrievalEngine
from egomemory.schema import MemoryEvent


PROJECT = Path(__file__).parent
INDEX_PATH = PROJECT / "results/fair_cooking_05_4/events.json"
VIDEO_PATH = PROJECT / "data/egoexo/takes/fair_cooking_05_4/frame_aligned_videos/downscaled/448/aria02_214-1.mp4"


@st.cache_resource
def load_system():
    events = [MemoryEvent(**event) for event in json.loads(INDEX_PATH.read_text(encoding="utf-8"))]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return events, MultimodalRetrievalEngine(events), model, processor, device


def embed_query(query: str, model, processor, device) -> list[float]:
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        output = model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        return model.text_projection(output.pooler_output)[0].cpu().numpy().tolist()


st.set_page_config(page_title="EgoMemory", page_icon="🧠", layout="wide")
st.title("🧠 EgoMemory")
st.caption("Multimodal episodic retrieval over a first-person cooking recording")

if not INDEX_PATH.exists() or not VIDEO_PATH.exists():
    st.error("The cooking-take index or video is missing. Build the index before launching the app.")
    st.stop()

examples = [
    "When did I touch the boiling noodles with my fingers?",
    "When did I say to drain the noodles?",
    "When did I explain that the heat of garlic comes from its germ?",
    "When was I shaking the pan?",
]
query = st.text_input("Ask about the cooking recording", value=examples[0], placeholder="e.g. When did I chop garlic?")

if st.button("Search memories", type="primary") and query.strip():
    events, engine, model, processor, device = load_system()
    results = engine.retrieve(embed_query(query, model, processor, device), query, k=5)
    st.session_state["query"] = query
    st.session_state["results"] = results

if "results" in st.session_state:
    st.subheader(f"Results for: {st.session_state['query']}")
    results = st.session_state["results"]
    labels = [f"{index + 1}. {result.event.start_time:.1f}–{result.event.end_time:.1f}s" for index, result in enumerate(results)]
    chosen_label = st.radio("Choose a retrieved memory", labels, horizontal=True)
    result = results[labels.index(chosen_label)]
    event = result.event
    scores = result.component_scores

    left, right = st.columns([3, 2])
    with left:
        st.video(str(VIDEO_PATH), start_time=int(event.start_time))
        st.caption(f"Playing from {event.start_time:.1f}s; selected window ends at {event.end_time:.1f}s.")
    with right:
        st.metric("Combined retrieval score", f"{result.score:.3f}")
        st.write("**Evidence used**")
        st.progress(max(0.0, min(1.0, scores["vision"])), text=f"Vision / FAISS: {scores['vision']:.3f}")
        st.progress(max(0.0, min(1.0, scores["transcript_audio"])), text=f"Audio transcript: {scores['transcript_audio']:.3f}")
        st.progress(max(0.0, min(1.0, scores["motion"])), text=f"Motion / trajectory: {scores['motion']:.3f}")
        st.write("**Aligned transcript**")
        st.info(event.narration or "No spoken transcript overlaps this window.")
        st.write("**Raw sensor summaries**")
        st.json({"audio_rms": event.audio_features.get("rms", 0.0), "motion_energy": event.motion_features.get("motion_energy", 0.0)})
else:
    st.info("Try an example query, then select **Search memories**.")
