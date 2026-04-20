"""
chatbot_engine.py

Core system:
- Emotion detection
- Context detection
- Decision engine
- Response system
- DB storage
"""

import random
from datetime import datetime

from emotion_model import EmotionModel
from database import Database


# ================================
# CHATBOT ENGINE
# ================================

class ChatbotEngine:
    def __init__(self):
        self.model = EmotionModel()

        # Load existing model — or train + save on first run
        try:
            self.model.load_model()
        except FileNotFoundError:
            print("[WARN] No trained model found. Training now...")
            self.model.train()
            self.model.save_model()

        self.db = Database()

        # Track last responses to avoid repetition
        self.last_responses = []

        # ================================
        # RESPONSE DATABASE (IMPORTANT)
        # ================================

        self.responses = {
            "happy": [
                "That's great to hear!",
                "I'm glad you're feeling good!",
                "That sounds wonderful 😊",
                "Keep that positive energy going!",
                "Love the vibe!",
                "That's awesome!",
                "You deserve this happiness!",
                "Nice, keep it up!"
            ],
            "sad": [
                "I'm sorry you're feeling this way.",
                "That sounds really tough.",
                "I'm here with you.",
                "It's okay to feel like this.",
                "Do you want to talk more about it?",
                "You're not alone in this.",
                "That must be hard.",
                "I understand, tell me more."
            ],
            "angry": [
                "That sounds frustrating.",
                "I can sense your anger.",
                "Do you want to vent it out?",
                "That situation seems unfair.",
                "Take a deep breath—I'm here.",
                "Want to talk about what triggered this?",
                "That's really intense.",
                "Let it out."
            ],
            "anxious": [
                "That sounds overwhelming.",
                "Take it one step at a time.",
                "You're doing your best.",
                "Try to breathe slowly.",
                "Want to break it down together?",
                "You're not alone in this.",
                "That must feel stressful.",
                "I'm here to help you through it."
            ],
            "neutral": [
                "Got it.",
                "Okay, tell me more.",
                "I'm listening.",
                "Alright.",
                "Go on.",
                "Hmm, interesting.",
                "I see.",
                "Continue."
            ]
        }

    # ================================
    # CONTEXT DETECTION
    # ================================
    def detect_context(self, text):
        """
        Detect simple context using keywords
        """

        text = text.lower()

        if any(word in text for word in ["exam", "test", "study"]):
            return "exam"

        if any(word in text for word in ["work", "job", "office"]):
            return "work"

        if any(word in text for word in ["friend", "family", "relationship"]):
            return "social"

        return "general"

    # ================================
    # DECISION ENGINE
    # ================================
    def decide_response_type(self, emotion, intensity):
        """
        Decide type based on emotion + intensity
        """

        if emotion in ["sad", "anxious"]:
            if intensity > 0.7:
                return "empathy"
            else:
                return "validation"

        if emotion == "angry":
            return "calming"

        if emotion == "happy":
            return "reinforcement"

        return "neutral"

    # ================================
    # RESPONSE GENERATOR
    # ================================
    def generate_response(self, emotion):
        """
        Avoid repeating recent responses
        """

        options = self.responses.get(emotion, self.responses["neutral"])

        # Avoid repetition
        available = [r for r in options if r not in self.last_responses]

        if not available:
            self.last_responses = []
            available = options

        response = random.choice(available)

        self.last_responses.append(response)

        # Keep memory small
        if len(self.last_responses) > 5:
            self.last_responses.pop(0)

        return response

    # ================================
    # MAIN CHAT FUNCTION
    # ================================
    def process_input(self, user_id, text):
        """
        Full pipeline execution.
        Handles both small-talk (greetings etc.) and emotion-bearing messages.
        """

        # 1. Emotion detection (includes small-talk check)
        result = self.model.predict(text)
        emotion = result["emotion"]
        intensity = result["intensity"]
        confidence = result.get("confidence", 1.0)
        is_small_talk = result.get("small_talk", False)

        # 2. Context detection
        context = self.detect_context(text)

        # 3. Response — use the model's small-talk reply if available
        if is_small_talk and result.get("response"):
            response = result["response"]
            response_type = "small_talk"
        else:
            response_type = self.decide_response_type(emotion, intensity)
            response = self.generate_response(emotion)

        # 4. Save to database (only non-small-talk turns affect analysis)
        if not is_small_talk:
            self.db.save_chat(
                user_id=user_id,
                message=text,
                emotion=emotion,
                intensity=intensity,
                context=context
            )

        return {
            "response":   response,
            "emotion":    emotion,
            "intensity":  intensity,
            "confidence": confidence,
            "context":    context,
            "type":       response_type,
        }


# ================================
# TEST BLOCK
# ================================

if __name__ == "__main__":
    print("Chatbot starting...")

    engine = ChatbotEngine()
    user_id = "test_user"

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print("Exiting chat...")
            break

        output = engine.process_input(user_id, user_input)

        print(f"Bot: {output['response']}")
        print(f"(Emotion: {output['emotion']}, Intensity: {output['intensity']}, Context: {output['context']})")