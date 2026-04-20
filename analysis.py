"""
analysis.py

Handles:
- Data analysis using Pandas & NumPy
- Visualization using Matplotlib
- Pattern detection
- Risk level estimation
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # safe default; charts are embedded via FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import Counter
from datetime import datetime
import tkinter as tk

from database import Database


# ================================
# ANALYSIS CLASS
# ================================

class EmotionAnalyzer:
    def __init__(self):
        self.db = Database()

    # ================================
    # LOAD DATA → DATAFRAME
    # ================================
    def load_user_data(self, user_id):
        chats = self.db.get_user_chats(user_id)

        if not chats:
            return None

        df = pd.DataFrame(chats)

        # Convert timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    # ================================
    # NUMERICAL ANALYSIS
    # ================================
    def compute_metrics(self, df):
        """
        Returns key stats using NumPy
        """

        avg_intensity = np.mean(df["intensity"])

        emotion_counts = df["emotion"].value_counts().to_dict()

        # Emotional stability:
        # lower variance = more stable
        stability = 1 - np.var(df["intensity"])

        return {
            "avg_intensity": round(avg_intensity, 2),
            "emotion_counts": emotion_counts,
            "stability_score": round(max(0, stability), 2)
        }

    # ================================
    # VISUALIZATION
    # ================================
    def plot_emotion_distribution(self, df):
        """
        Open a Tkinter Toplevel with an embedded bar chart.
        Does not block the main event loop.
        """
        counts = df["emotion"].value_counts()

        # Build figure using the OO API (no pyplot)
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        colours = ["#2E86AB", "#E27D60", "#41B3A3", "#C38D9E", "#E8A87C"]
        bars = ax.bar(
            [e.capitalize() for e in counts.index],
            counts.values,
            color=(colours * 3)[:len(counts)],
            edgecolor="white",
        )
        ax.set_title("Emotion Distribution", fontsize=13, fontweight="bold")
        ax.set_ylabel("Frequency")
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    str(val), ha="center", va="bottom", fontsize=9)
        fig.tight_layout()

        # Embed in Tkinter Toplevel
        win = tk.Toplevel()
        win.title("Emotion Distribution")
        win.configure(bg="#1B1B2F")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_emotion_trend(self, df):
        """
        Open a Tkinter Toplevel with an embedded line chart.
        Does not block the main event loop.
        """
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        indices = list(range(1, len(df_sorted) + 1))
        intensities = df_sorted["intensity"].tolist()

        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(indices, intensities, color="#2E86AB", linewidth=2,
                marker="o", markersize=4, markerfacecolor="#E27D60",
                markeredgecolor="white", markeredgewidth=0.8)
        ax.fill_between(indices, intensities, alpha=0.10, color="#2E86AB")
        ax.set_title("Emotion Intensity Over Time", fontsize=13, fontweight="bold")
        ax.set_xlabel("Message #")
        ax.set_ylabel("Intensity")
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()

        win = tk.Toplevel()
        win.title("Intensity Trend")
        win.configure(bg="#1B1B2F")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ================================
    # PATTERN DETECTION
    # ================================
    def detect_patterns(self, df):
        patterns = {}

        # Most frequent emotion
        patterns["dominant_emotion"] = df["emotion"].mode()[0]

        # Repeated emotion detection
        patterns["repeated_negative"] = (
            df["emotion"].isin(["sad", "angry", "anxious"]).sum()
        )

        # Trigger words (simple keyword extraction)
        all_words = " ".join(df["message"]).lower().split()
        common_words = Counter(all_words).most_common(5)

        patterns["top_keywords"] = common_words

        # Time-based behavior (night stress)
        df["hour"] = df["timestamp"].dt.hour
        night_entries = df[df["hour"] >= 22]

        patterns["night_stress"] = len(night_entries)

        return patterns

    # ================================
    # RISK LEVEL
    # ================================
    def calculate_risk(self, df):
        """
        Determine risk level based on:
        - negative emotion frequency
        - intensity
        """

        negative_df = df[df["emotion"].isin(["sad", "angry", "anxious"])]

        ratio = len(negative_df) / len(df)
        avg_intensity = np.mean(df["intensity"])

        if ratio > 0.6 and avg_intensity > 0.7:
            return "High", "User shows elevated emotional stress patterns over recent sessions."

        elif ratio > 0.4:
            return "Moderate", "User shows moderate emotional fluctuations."

        else:
            return "Low", "User emotional patterns appear relatively stable."

    # ================================
    # FULL ANALYSIS PIPELINE
    # ================================
    def analyze_user(self, user_id):
        df = self.load_user_data(user_id)

        if df is None:
            return {"error": "No data found"}

        metrics = self.compute_metrics(df)
        patterns = self.detect_patterns(df)
        risk_level, message = self.calculate_risk(df)

        return {
            "metrics": metrics,
            "patterns": patterns,
            "risk_level": risk_level,
            "message": message
        }


# ================================
# TEST BLOCK
# ================================
if __name__ == "__main__":
    analyzer = EmotionAnalyzer()

    user_id = input("Enter user_id: ")

    result = analyzer.analyze_user(user_id)

    print("\n--- ANALYSIS ---")
    print(result)

    # Optional plots
    df = analyzer.load_user_data(user_id)

    if df is not None:
        analyzer.plot_emotion_distribution(df)
        analyzer.plot_emotion_trend(df)