"""Native Windows desktop interface for EgoMemory.

Run from the project root with: run_desktop.cmd
"""

import json
import csv
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
from transformers import CLIPModel, CLIPProcessor

from egomemory.retrieval.multimodal import MultimodalRetrievalEngine
from egomemory.schema import MemoryEvent


ROOT = Path(__file__).parent
EVENTS_PATH = ROOT / "results/fair_cooking_05_4/events.json"
VIDEO_PATH = ROOT / "data/egoexo/takes/fair_cooking_05_4/frame_aligned_videos/downscaled/448/aria02_214-1.mp4"
ABLATION_PATH = ROOT / "results/ablation_50_provisional/summary.csv"


class EgoMemoryApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EgoMemory — Multimodal Cooking Memory")
        self.geometry("1180x760")
        self.minsize(920, 620)
        self.results = []
        self.capture = None
        self.playing = False
        self.photo = None

        events = [MemoryEvent(**event) for event in json.loads(EVENTS_PATH.read_text(encoding="utf-8"))]
        self.engine = MultimodalRetrievalEngine(events)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.ablation_summary = list(csv.DictReader(ABLATION_PATH.open(encoding="utf-8"))) if ABLATION_PATH.exists() else []
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self):
        search = ttk.Frame(self, padding=14)
        search.pack(fill="x")
        ttk.Label(search, text="Ask about the cooking recording:", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        row = ttk.Frame(search)
        row.pack(fill="x", pady=(6, 0))
        self.query = tk.StringVar(value="When did I explain that the heat of garlic comes from its germ?")
        entry = ttk.Entry(row, textvariable=self.query, font=("Segoe UI", 12))
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda _event: self.search())
        ttk.Button(row, text="Search memories", command=self.search).pack(side="left", padx=(8, 0))

        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        left = ttk.Frame(body, padding=8)
        right = ttk.Frame(body, padding=8)
        body.add(left, weight=1)
        body.add(right, weight=2)

        ttk.Label(left, text="Retrieved memories", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.listbox = tk.Listbox(left, font=("Segoe UI", 10), exportselection=False)
        self.listbox.pack(fill="both", expand=True, pady=(6, 0))
        self.listbox.bind("<<ListboxSelect>>", self.select_result)

        self.video_label = ttk.Label(right, text="Search to see a video preview", anchor="center")
        self.video_label.pack(fill="both", expand=True)
        controls = ttk.Frame(right)
        controls.pack(fill="x", pady=8)
        self.play_button = ttk.Button(controls, text="Play selected window", command=self.toggle_playback, state="disabled")
        self.play_button.pack(side="left")
        self.time_label = ttk.Label(controls, text="")
        self.time_label.pack(side="left", padx=12)
        ttk.Label(right, text="Modality evidence", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(5, 0))
        self.evidence = ttk.Label(right, text="", justify="left", font=("Segoe UI", 10))
        self.evidence.pack(anchor="w", fill="x", pady=(4, 8))
        ttk.Label(right, text="Aligned audio transcript", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.transcript = tk.Text(right, height=6, wrap="word", font=("Segoe UI", 10), state="disabled")
        self.transcript.pack(fill="x", pady=(4, 0))
        evaluation = ttk.LabelFrame(right, text="Retrieval evaluation — fixed weights, 50 provisional labels", padding=6)
        evaluation.pack(fill="x", pady=(10, 0))
        self.evaluation_table = ttk.Treeview(evaluation, columns=("system", "recall", "error"), show="headings", height=7)
        self.evaluation_table.heading("system", text="System")
        self.evaluation_table.heading("recall", text="Recall@5")
        self.evaluation_table.heading("error", text="Avg. temporal error")
        self.evaluation_table.column("system", width=150)
        self.evaluation_table.column("recall", width=80, anchor="center")
        self.evaluation_table.column("error", width=130, anchor="center")
        self.evaluation_table.pack(fill="x")
        for row in self.ablation_summary:
            self.evaluation_table.insert("", "end", values=(row["configuration"].replace("_", " + "), f"{float(row['recall_at_5']):.1%}", f"{float(row['mean_temporal_error_seconds']):.1f}s"))

    def embed_query(self, query):
        inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            output = self.model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            return self.model.text_projection(output.pooler_output)[0].cpu().numpy().tolist()

    def search(self):
        text = self.query.get().strip()
        if not text:
            return
        self.results = self.engine.retrieve(self.embed_query(text), text, k=5)
        self.listbox.delete(0, tk.END)
        for rank, result in enumerate(self.results, start=1):
            event = result.event
            self.listbox.insert(tk.END, f"{rank}. {event.start_time:06.1f}–{event.end_time:06.1f}s")
        self.listbox.selection_set(0)
        self.show_result(0)

    def select_result(self, _event=None):
        selection = self.listbox.curselection()
        if selection:
            self.show_result(selection[0])

    def show_result(self, index):
        self.playing = False
        self.play_button.configure(text="Play selected window", state="normal")
        result = self.results[index]
        event = result.event
        scores = result.component_scores
        self.window_start, self.window_end = event.start_time, event.end_time
        self.show_frame(self.window_start)
        self.time_label.configure(text=f"{event.start_time:.1f}s – {event.end_time:.1f}s")
        self.evidence.configure(text=(
            f"Combined score: {result.score:.3f}\n"
            f"Vision / FAISS: {scores['vision']:.3f}\n"
            f"Audio transcript: {scores['transcript_audio']:.3f}\n"
            f"Motion / trajectory: {scores['motion']:.3f}\n"
            f"Raw audio activity: {event.audio_features.get('rms', 0.0):.4f}"
        ))
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", tk.END)
        self.transcript.insert("1.0", event.narration or "No spoken transcript overlaps this window.")
        self.transcript.configure(state="disabled")

    def show_frame(self, seconds):
        if self.capture is None:
            self.capture = cv2.VideoCapture(str(VIDEO_PATH))
        self.capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = self.capture.read()
        if not ok:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image.thumbnail((730, 440))
        self.photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self.photo, text="")

    def toggle_playback(self):
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play selected window")
        if self.playing:
            self.play_position = self.window_start
            self.play_next()

    def play_next(self):
        if not self.playing:
            return
        if self.play_position >= self.window_end:
            self.playing = False
            self.play_button.configure(text="Replay selected window")
            return
        self.show_frame(self.play_position)
        self.play_position += 0.1
        self.after(100, self.play_next)

    def close(self):
        if self.capture is not None:
            self.capture.release()
        self.destroy()


if __name__ == "__main__":
    if not EVENTS_PATH.exists() or not VIDEO_PATH.exists():
        raise SystemExit("Build the cooking-take index before launching the desktop app.")
    EgoMemoryApp().mainloop()
