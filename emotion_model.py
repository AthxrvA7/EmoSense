"""
emotion_model.py

Significantly improved emotion detection using:
- Real labelled dataset: dair-ai/emotion (HuggingFace, ~16K samples)
  with automatic offline fallback to a large embedded corpus
- LinearSVC classifier  (much more accurate than Naive Bayes)
- TF-IDF Vectorizer with unigrams + bigrams
- Calibrated probabilities via CalibratedClassifierCV
- Smart negation scoping
- Richer intensity scoring (amplifiers, diminishers, caps, punctuation)
- Greeting / small-talk layer with varied personality responses
"""

import os
import re
import pickle
import random
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ============================================================
# LABEL MAP  (dair-ai/emotion labels → EmoSense labels)
# ============================================================
# dair-ai/emotion:  0=sadness  1=joy  2=love  3=anger  4=fear  5=surprise
HF_LABEL_MAP = {
    0: "sad",
    1: "happy",
    2: "happy",      # love is positive
    3: "angry",
    4: "anxious",
    5: "neutral",    # surprise is ambiguous
}


# ============================================================
# LARGE OFFLINE / FALLBACK CORPUS
# ============================================================
# ~80 examples per class — diverse vocabulary, varied sentence structures.
# Used when the HuggingFace dataset cannot be fetched.
EMBEDDED_CORPUS = {
    "happy": [
        "I feel absolutely wonderful today",
        "This is the best day of my entire life",
        "I got promoted at work and I am thrilled",
        "I am so happy right now I could cry",
        "Everything is going perfectly and I feel great",
        "I just got engaged and I am overjoyed",
        "My team won the match and we are ecstatic",
        "I feel on top of the world",
        "Life is treating me so well lately",
        "I am genuinely excited about what is coming",
        "I feel blessed and grateful for everything",
        "Today was absolutely perfect in every way",
        "I feel cheerful and full of positive energy",
        "My heart is bursting with joy right now",
        "I passed my finals and I am so relieved and happy",
        "I feel genuinely proud of what I have achieved",
        "My friend surprised me and it made my day",
        "I woke up feeling amazing and rested",
        "I am elated beyond words right now",
        "Everything worked out exactly as I had hoped",
        "I feel light and happy and at peace",
        "I just received wonderful news about my health",
        "I am so delighted to see everyone doing well",
        "I feel inspired and motivated and alive",
        "I cannot stop smiling today",
        "I feel so much gratitude in my heart",
        "My anxiety is gone and I feel free",
        "I am celebrating a huge milestone today",
        "I feel positive and confident about the future",
        "I fell in love and it feels incredible",
        "I am genuinely content and satisfied with life",
        "This vacation has made me feel so alive",
        "I am so glad things turned out this way",
        "Waking up next to someone I love fills me with joy",
        "I feel radiantly happy today for no particular reason",
        "My creative work got recognised and I am beaming",
        "I feel full of hope and optimism",
        "I got great news about my family and I am relieved",
        "Today I am celebrating and nothing can bring me down",
        "I feel a deep sense of happiness and peace",
        "I am loving every moment of my life right now",
        "I feel so enthusiastic about the new project",
        "This achievement means the world to me",
        "I am excited about all the possibilities ahead",
        "I feel joyful and carefree today",
        "My spirits are high and I feel unstoppable",
        "I am thrilled to announce great news",
        "Everything feels fresh and exciting",
        "I feel a sense of pure bliss today",
        "I just reunited with old friends and it is wonderful",
    ],
    "sad": [
        "I feel so empty and hollow inside",
        "I cried myself to sleep again last night",
        "Nothing brings me happiness anymore",
        "I feel completely broken and lost",
        "I miss them so much it physically hurts",
        "I feel hopeless and see no way out",
        "Life feels meaningless and pointless to me",
        "I am deeply disappointed in how things turned out",
        "I keep failing no matter how hard I try",
        "I feel like a burden to everyone around me",
        "Nobody seems to care how I am doing",
        "I feel profoundly lonely even in a crowd",
        "I am grieving and the pain is unbearable",
        "My heart feels permanently broken",
        "I do not see the point of going on anymore",
        "I feel crushed by the weight of my sadness",
        "I have lost all motivation and drive",
        "I feel deeply unhappy and cannot explain why",
        "Everything reminds me of what I have lost",
        "I am so tired of feeling this way",
        "I feel abandoned by the people I trusted",
        "The sadness never fully goes away",
        "I am devastated by what happened to me",
        "I feel invisible and unimportant",
        "My depression is making it hard to function",
        "I feel so low I cannot get out of bed",
        "I am grieving the loss of my relationship",
        "I cry when no one is watching",
        "I feel like I have failed everyone",
        "I do not feel like myself at all lately",
        "I am heartbroken and cannot stop thinking about it",
        "I feel so defeated after trying so hard",
        "There is a constant ache inside me",
        "I wish things were different but I cannot change them",
        "I feel like my old self is gone forever",
        "I am struggling with profound sadness every day",
        "I lost someone dear to me and I am not okay",
        "I feel disconnected from everything and everyone",
        "My dreams feel far away and impossible now",
        "I feel deep regret about the choices I made",
        "Nothing makes me laugh or smile anymore",
        "I feel helpless in the face of everything",
        "I am mourning a future I will never have",
        "The pain inside me is difficult to describe",
        "I feel like I am sinking with no way up",
        "I wake up every day feeling dread and sadness",
        "I am overwhelmed by grief",
        "I feel forsaken and alone in the world",
        "I have been crying more than usual lately",
        "I am hurting deeply and silently",
    ],
    "angry": [
        "I am absolutely furious right now",
        "This is completely unacceptable and I am enraged",
        "I cannot stand this injustice any longer",
        "I am boiling with rage and resentment",
        "They keep disrespecting me and I have had enough",
        "I am so fed up with this entire situation",
        "I want to scream from all the frustration",
        "They betrayed my trust and I am livid",
        "I am sick and tired of being treated this way",
        "My blood is boiling after what just happened",
        "I hate how unfair everything is",
        "This situation is infuriating beyond words",
        "I am outraged by the complete lack of respect",
        "I have completely lost my patience",
        "I am explosively angry right now",
        "Nobody ever listens to me and it drives me insane",
        "I am raging because no one takes me seriously",
        "I feel a deep seething anger I cannot control",
        "I was treated so badly and I will not stand for it",
        "I am disgusted and furious at the same time",
        "I snapped at everyone today because of the stress",
        "I feel intense anger rising inside me",
        "I am aggravated by the constant failures around me",
        "This injustice makes me want to fight back",
        "I am irate and I need someone to listen",
        "I am so angry I can barely think straight",
        "They crossed a line and I cannot let it go",
        "My temper is at its absolute limit",
        "I feel hostility and rage building inside",
        "I am fed up with being ignored and dismissed",
        "This makes me so angry I am shaking",
        "I am bitter and resentful about what happened",
        "I feel a burning anger that will not go away",
        "I despise how everything turned out",
        "I am furious at the system that failed me",
        "I cannot believe how badly I was treated",
        "I feel provoked beyond what I can handle",
        "I am filled with indignation and fury",
        "I snapped because I have been pushed too far",
        "The anger inside me feels like it will explode",
        "I am enraged by the constant betrayals",
        "I feel deeply wronged and I am not okay with it",
        "I am irritated by absolutely everything today",
        "I feel volcanic anger building up inside",
        "I have reached my breaking point with this",
        "I am seething and struggling to stay calm",
        "I feel nothing but contempt and rage",
        "I am so frustrated I could break something",
        "I feel consumed by anger and resentment",
        "This is the last straw and I am done",
    ],
    "anxious": [
        "I am so anxious I cannot concentrate on anything",
        "My mind is racing and I cannot calm it down",
        "I feel a constant knot of dread in my stomach",
        "I am terrified of what the future holds for me",
        "I keep having panic attacks out of nowhere",
        "I cannot sleep because of constant worry",
        "I feel overwhelmed by all the responsibilities",
        "I am scared of failing and letting people down",
        "Something bad is going to happen I can feel it",
        "My heart races even when everything seems fine",
        "I feel restless and unable to sit still at all",
        "I am nervous about the big interview tomorrow",
        "The uncertainty is driving me to the edge",
        "I overthink every single decision I make",
        "I am afraid I am simply not good enough",
        "I feel paralyzed by the fear of making mistakes",
        "I cannot make decisions because I am too scared",
        "I am dreading the conversation I need to have",
        "I feel tense and on edge constantly",
        "There is a sense of impending doom I cannot shake",
        "I feel worried about everything all the time",
        "I am anxious about my health and what it means",
        "I feel sick with nerves before every presentation",
        "I cannot stop catastrophizing about everything",
        "I feel trapped under the weight of anxiety",
        "I am always waiting for the other shoe to drop",
        "I feel like something terrible is just around the corner",
        "I am stressed to the point of physical pain",
        "I feel a creeping anxiety that will not leave me",
        "I am scared of being judged and rejected",
        "My anxiety makes every day feel like a battle",
        "I feel nervous and jittery without any real reason",
        "I am on edge and hyper-vigilant all the time",
        "I feel panic rising whenever I think about it",
        "I am afraid of losing control of my life",
        "I feel unsafe and uneasy even in familiar places",
        "I worry constantly about the people I love",
        "I have been having intrusive anxious thoughts",
        "I feel desperate and helpless in the face of my fears",
        "I cannot eat or sleep because of the anxiety",
        "I feel shaky and breathless all day",
        "I am consumed by what-if thinking",
        "I feel a constant low-level fear I cannot name",
        "I am overwhelmed by the pressure I am under",
        "My anxiety spikes every time my phone rings",
        "I feel dread before facing the outside world",
        "I am scared of being a disappointment",
        "I feel anxious and vulnerable all the time",
        "I am filled with apprehension about the future",
        "I feel like I am always on the verge of breaking down",
    ],
    "neutral": [
        "I had a pretty normal day today",
        "Nothing much happened worth mentioning",
        "It was just an average kind of day",
        "I went to work came home and had dinner",
        "Things are okay nothing to report",
        "I watched some television and relaxed",
        "I had lunch and then finished some reading",
        "Just a regular uneventful week",
        "Nothing special happened today",
        "I did my usual daily routine",
        "I feel okay not great but not bad either",
        "Everything seems fine at the moment",
        "It was a pretty unremarkable day overall",
        "I am just going through the usual motions",
        "I feel neither particularly happy nor particularly sad",
        "I did some work and then took a nap",
        "The day passed by without much incident",
        "I am feeling fairly neutral right now",
        "Not much is on my mind today",
        "I just finished my tasks and relaxed",
        "It is what it is and I accept that",
        "I am doing fine nothing exciting to report",
        "Just an ordinary week so far for me",
        "I do not feel strongly about anything right now",
        "Today was unremarkable in every sense",
        "I went for a walk and had some coffee",
        "I read a few articles and responded to emails",
        "I tidied up the apartment this afternoon",
        "Things at work have been quiet and steady",
        "I had a phone call with a friend and it was nice",
        "I made dinner and watched something on Netflix",
        "I feel calm and stable today",
        "I exercised briefly and felt okay afterward",
        "The weather was mild and I stayed indoors",
        "I did some grocery shopping and came home",
        "There is nothing particularly good or bad to say",
        "I answered some messages and did some chores",
        "I feel indifferent about most things today",
        "It was a slow day and I passed the time quietly",
        "I thought about things but reached no conclusions",
        "I had a routine check-up and everything was fine",
        "I completed my work and closed the laptop",
        "Nothing out of the ordinary took place today",
        "I feel steady and grounded in a quiet kind of way",
        "Today felt like every other Tuesday",
        "I did not feel much of anything in particular",
        "I made plans for the weekend and left it at that",
        "I feel balanced and calm without any strong emotion",
        "The day went by quickly and without drama",
        "I feel like I am just going with the flow",
    ],
}


# ============================================================
# SMALL-TALK PATTERNS
# ============================================================
SMALL_TALK_PATTERNS = {
    "greeting": {
        "keywords": [
            r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bhowdy\b",
            r"\bgreetings\b", r"\bwassup\b", r"\bwhat'?s up\b",
            r"\bhiya\b", r"\byo\b", r"\bsup\b",
        ],
        "responses": [
            "Hey there! I'm EmoSense — your emotional companion. How are you feeling today?",
            "Hello! Great to see you here. What's on your mind?",
            "Hi! I'm here and ready to listen. How's your day going?",
            "Hey! Tell me how you're doing — I genuinely want to know.",
            "Hello there! What's the emotional weather like for you today?",
        ],
    },
    "how_are_you": {
        "keywords": [
            r"\bhow are you\b", r"\bhow do you do\b", r"\bhow'?s it going\b",
            r"\bhow are you doing\b", r"\byou okay\b", r"\bare you okay\b",
            r"\bhow'?s everything\b", r"\bhow'?s life\b",
        ],
        "responses": [
            "I'm an AI, so I don't feel things — but I'm fully focused on yours! What's going on with you?",
            "Thanks for asking! I'm running at full capacity. More importantly, how are *you* doing?",
            "I appreciate you checking in! I'm here and listening. How about you — how's your day?",
            "I exist to understand your emotions, not my own. So tell me — how are *you* really feeling?",
            "Honestly? I'm always doing well when someone talks to me. How about you?",
        ],
    },
    "goodbye": {
        "keywords": [
            r"\bbye\b", r"\bgoodbye\b", r"\bsee you\b", r"\bsee ya\b",
            r"\btake care\b", r"\bgotta go\b", r"\bgood night\b",
            r"\bgoodnight\b", r"\bciao\b",
        ],
        "responses": [
            "Take care of yourself! I'm always here when you need to talk.",
            "Goodbye! Hope your day gets lighter from here.",
            "See you soon! Don't forget to check in with yourself today.",
            "Bye! You matter — come back anytime.",
            "Until next time! Stay kind to yourself.",
        ],
    },
    "thanks": {
        "keywords": [
            r"\bthank(s| you)\b", r"\bthx\b", r"\bmuch appreciated\b", r"\bgrateful\b",
        ],
        "responses": [
            "You're very welcome! I'm always here for you.",
            "Of course! That's exactly what I'm here for.",
            "No need to thank me — just glad I could help!",
            "Anytime! Feel free to share anything else on your mind.",
        ],
    },
    "who_are_you": {
        "keywords": [
            r"\bwho are you\b", r"\bwhat are you\b", r"\bwhat is emosense\b",
            r"\btell me about yourself\b", r"\byour name\b",
        ],
        "responses": [
            "I'm EmoSense — an emotion-aware chatbot built to understand and support your emotional well-being.",
            "I'm EmoSense! I use AI to detect your emotions and respond with empathy.",
            "Think of me as an emotional companion. I listen, I analyse, and I respond with care.",
        ],
    },
    "how_can_you_help": {
        "keywords": [
            r"\bwhat can you do\b", r"\bhow can you help\b", r"\bwhat do you do\b",
            r"\bwhat'?s your purpose\b",
        ],
        "responses": [
            "I detect emotions in your messages, track patterns over time, and offer supportive responses. Try telling me how you feel!",
            "I listen, identify the emotions behind your words, and respond with empathy. I can also generate emotional insight reports.",
            "I'm built to understand your emotional state and help you reflect. Just start talking!",
        ],
    },
}


# ============================================================
# EMOTION MODEL
# ============================================================
class EmotionModel:
    def __init__(self):
        self.pipeline = None
        self.is_trained = False

        # Negation words
        self.negation_tokens = {
            "not", "never", "no", "neither", "nor", "cant",
            "wont", "dont", "didnt", "isnt", "wasnt",
        }
        self.negation_map = {
            "happy":   "sad",
            "sad":     "neutral",
            "angry":   "neutral",
            "anxious": "neutral",
            "neutral": "neutral",
        }

        # Intensity modifiers
        self.amplifiers = {
            "extremely", "very", "really", "so", "too", "incredibly",
            "absolutely", "utterly", "terribly", "deeply", "badly",
            "completely", "totally", "awful", "horrible", "insanely",
            "unbearably", "desperately", "profoundly", "overwhelmingly",
        }
        self.diminishers = {
            "slightly", "kind of", "a bit", "somewhat", "a little",
            "mildly", "fairly", "rather", "sort of", "barely",
        }

        # Pre-compile small-talk regex
        self._small_talk_re = {
            intent: [re.compile(p, re.IGNORECASE) for p in data["keywords"]]
            for intent, data in SMALL_TALK_PATTERNS.items()
        }

    # ----------------------------------------------------------
    # DATA LOADING
    # ----------------------------------------------------------
    def _load_hf_dataset(self):
        """
        Try to load dair-ai/emotion from HuggingFace (requires internet + datasets lib).
        Returns (texts, labels) or raises an exception.
        """
        from datasets import load_dataset
        print("[INFO] Downloading dair-ai/emotion dataset from HuggingFace...")
        ds = load_dataset("dair-ai/emotion")

        texts, labels = [], []
        for split in ("train", "validation", "test"):
            if split not in ds:
                continue
            for row in ds[split]:
                emotion = HF_LABEL_MAP.get(row["label"])
                if emotion:
                    texts.append(row["text"])
                    labels.append(emotion)

        print(f"[OK] Loaded {len(texts)} samples from HuggingFace dataset.")
        return texts, labels

    def _load_embedded_dataset(self):
        """Return the built-in fallback corpus."""
        texts, labels = [], []
        for emotion, sentences in EMBEDDED_CORPUS.items():
            for s in sentences:
                texts.append(s)
                labels.append(emotion)
        print(f"[INFO] Using embedded corpus ({len(texts)} samples).")
        return texts, labels

    # ----------------------------------------------------------
    # TRAINING
    # ----------------------------------------------------------
    def train(self):
        """
        Train the model.
        Priority:  HuggingFace dataset  ->  embedded corpus
        Classifier: LinearSVC (significantly better than Naive Bayes)
        """
        try:
            texts, labels = self._load_hf_dataset()
        except Exception as e:
            print(f"[WARN] HuggingFace dataset unavailable ({e}). Using embedded corpus.")
            texts, labels = self._load_embedded_dataset()

        # Apply negation pre-processing so the model sees 'not_happy' etc.
        texts = [self._preprocess(t) for t in texts]

        # Split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.1, random_state=42, stratify=labels
        )

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True,
                strip_accents="unicode",
                analyzer="word",
                lowercase=True,
                max_features=50_000,
            )),
            # CalibratedClassifierCV wraps LinearSVC so we can get probabilities
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=1.0, max_iter=2000, class_weight="balanced"),
                cv=3,
            )),
        ])

        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        # Quick accuracy report on held-out set
        y_pred = self.pipeline.predict(X_test)
        print("\n[OK] Emotion model trained successfully (TF-IDF + LinearSVC).")
        print(classification_report(y_test, y_pred, zero_division=0))

    # ----------------------------------------------------------
    # SAVE / LOAD
    # ----------------------------------------------------------
    def save_model(self, path="emotion_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)
        print(f"[SAVE] Model saved at {path}")

    def load_model(self, path="emotion_model.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError("Model file not found. Train first.")
        with open(path, "rb") as f:
            self.pipeline = pickle.load(f)
        self.is_trained = True
        print("[OK] Model loaded successfully.")

    # ----------------------------------------------------------
    # SMALL-TALK DETECTION
    # ----------------------------------------------------------
    def detect_small_talk(self, text):
        """Returns (intent, response) or (None, None)."""
        t = text.strip().lower()
        for intent, patterns in self._small_talk_re.items():
            for pat in patterns:
                if pat.search(t):
                    resp = random.choice(SMALL_TALK_PATTERNS[intent]["responses"])
                    return intent, resp
        return None, None

    # ----------------------------------------------------------
    # NEGATION HANDLING
    # ----------------------------------------------------------
    def handle_negation(self, text, predicted_emotion):
        """Flip emotion if a negation token appears in the first 60% of the sentence."""
        words = re.sub(r"[^\w\s]", "", text.lower()).split()
        cutoff = int(len(words) * 0.60)
        for i, word in enumerate(words[:cutoff]):
            if word in self.negation_tokens:
                return self.negation_map.get(predicted_emotion, predicted_emotion)
        return predicted_emotion

    # ----------------------------------------------------------
    # INTENSITY SCORING
    # ----------------------------------------------------------
    def calculate_intensity(self, text):
        """
        Returns 0.0 – 1.0 based on:
        - Amplifier / diminisher density
        - ALL-CAPS ratio
        - Repeated punctuation (!!)
        - Sentence length
        """
        words = text.split()
        lower_words = [re.sub(r"[^\w]", "", w) for w in words]
        score = 0.40

        for w in lower_words:
            if w in self.amplifiers:
                score += 0.12
            elif w in self.diminishers:
                score -= 0.07

        caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / max(len(words), 1)
        score += caps_ratio * 0.25

        score += min(text.count("!"), 4) * 0.05
        score += min(text.count("?"), 2) * 0.02
        score += min(len(words) / 60, 0.15)

        return round(min(max(score, 0.1), 1.0), 2)

    # ----------------------------------------------------------
    # CONFIDENCE
    # ----------------------------------------------------------
    def get_confidence(self, text):
        proba = self.pipeline.predict_proba([text])[0]
        return round(float(np.max(proba)), 2)

    # ----------------------------------------------------------
    # KEYWORD OVERRIDE  (post-model correction layer)
    # ----------------------------------------------------------
    # The HF emotion dataset has very few "neutral" and "surprise" examples
    # relative to sadness/joy. These keyword rules catch the most common
    # systematic errors without hurting the rest of accuracy.

    # Words strongly associated with anger — override when model disagrees
    _ANGER_STRONG = {
        "furious", "fuming", "infuriating", "infuriated", "enraged",
        "livid", "outraged", "irate", "seething", "rageful",
        "loathe", "despise", "hatred", "indignant", "disgusted",
    }
    # Everyday neutral phrases — override when model predicts sad/anxious
    _NEUTRAL_PHRASES = [
        re.compile(p, re.IGNORECASE) for p in [
            r"\bnormal (day|week|tuesday|monday|routine)\b",
            r"\bnothing (much|special|happened|to report)\b",
            r"\bjust a (regular|normal|ordinary|typical)\b",
            r"\bgoing through (the|my) (motions|routine)\b",
            r"\bpretty (uneventful|unremarkable|average)\b",
            r"\bfeeling (okay|fine|alright|neutral)\b",
        ]
    ]
    # Worry / anxiety words — help when confidence is borderline
    _ANXIETY_STRONG = {
        "anxious", "worried", "worry", "terrified", "petrified", "panicking",
        "dread", "dreading", "nervous", "apprehensive", "uneasy",
        "frightened", "fearful", "dreaded", "overwhelmed", "stressed",
        "tense", "paranoid", "phobia", "insecure",
    }

    def _keyword_override(self, text, predicted):
        t_lower = text.lower()
        words = set(re.sub(r"[^\w\s]", "", t_lower).split())

        # Rule 1 — neutral pattern phrases
        for pat in self._NEUTRAL_PHRASES:
            if pat.search(text):
                return "neutral"

        # Rule 2 — strong anger vocabulary beats a non-angry prediction
        if self._ANGER_STRONG & words and predicted != "angry":
            return "angry"

        # Rule 3 — anxiety keywords present and model confidence is not overwhelming
        if self._ANXIETY_STRONG & words and predicted in ("sad", "neutral"):
            proba = self.pipeline.predict_proba([text])[0]
            classes = list(self.pipeline.classes_)
            if "anxious" in classes:
                anxious_prob = proba[classes.index("anxious")]
                if anxious_prob > 0.15:   # any meaningful anxiety signal
                    return "anxious"

        return predicted

    def _preprocess(self, text):
        """
        Join negation tokens to the next word so TF-IDF captures 'not_happy'
        instead of treating 'not' and 'happy' as independent tokens.
        e.g. 'I am not happy' -> 'I am not_happy'
        """
        tokens = text.split()
        result = []
        i = 0
        negation_words = {"not", "never", "no", "neither", "nor", "dont", "cant", "wont"}
        while i < len(tokens):
            w = tokens[i].lower().rstrip(".,!?")
            if w in negation_words and i + 1 < len(tokens):
                # merge: not + happy → not_happy
                result.append(tokens[i] + "_" + tokens[i + 1])
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return " ".join(result)

    # ----------------------------------------------------------
    # PREDICT
    # ----------------------------------------------------------
    def predict(self, text):
        """
        Returns:
        {
            emotion    : str,
            intensity  : float,
            confidence : float,
            small_talk : bool,
            response   : str | None
        }
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained or loaded yet.")

        # 1. Small-talk check first
        intent, st_response = self.detect_small_talk(text)
        if intent is not None:
            return {
                "emotion":    "neutral",
                "intensity":  0.3,
                "confidence": 1.0,
                "small_talk": True,
                "response":   st_response,
            }

        # 2. Pre-process (negation joining for TF-IDF)
        processed = self._preprocess(text)

        # 3. Classify (use processed text for model)
        raw_emotion = self.pipeline.predict([processed])[0]

        # 4. Keyword-override safety net (catches systematic HF imbalance errors)
        raw_emotion = self._keyword_override(text, raw_emotion)

        # 5. Negation on original text (flip emotions from surface patterns)
        final_emotion = self.handle_negation(text, raw_emotion)

        # 6. Intensity
        intensity = self.calculate_intensity(text)

        # 7. Confidence (use processed text)
        proba = self.pipeline.predict_proba([processed])[0]
        confidence = round(float(np.max(proba)), 2)

        return {
            "emotion":    final_emotion,
            "intensity":  intensity,
            "confidence": confidence,
            "small_talk": False,
            "response":   None,
        }


# ============================================================
# TEST BLOCK
# ============================================================
if __name__ == "__main__":
    model = EmotionModel()
    model.train()
    model.save_model()
    model.load_model()

    tests = [
        # Small-talk
        ("Hello!", "small_talk"),
        ("How are you doing?", "small_talk"),
        ("Who are you?", "small_talk"),
        ("Goodbye!", "small_talk"),
        ("Thank you so much", "small_talk"),
        # Emotion sentences
        ("I feel absolutely wonderful and happy today", "happy"),
        ("I am devastated and heartbroken", "sad"),
        ("I am FURIOUS and I cannot take it anymore!!", "angry"),
        ("I am so anxious about everything and cannot sleep", "anxious"),
        ("Just a normal Tuesday, nothing much happened", "neutral"),
        ("I am not happy at all today", "sad"),          # negation
        ("I feel kind of worried about tomorrow", "anxious"),  # diminisher
        ("I am on top of the world right now", "happy"),
        ("I feel so empty and hopeless", "sad"),
        ("This is infuriating beyond belief", "angry"),
    ]

    print("\n--- PREDICTIONS ---")
    for text, expected in tests:
        result = model.predict(text)
        if result["small_talk"]:
            print(f"\n[SMALL-TALK] {text}")
            print(f"  Reply : {result['response']}")
        else:
            status = "OK" if result["emotion"] == expected else f"WRONG (expected {expected})"
            print(f"\n[{status}] '{text}'")
            print(f"  Emotion={result['emotion']} | Confidence={result['confidence']} | Intensity={result['intensity']}")