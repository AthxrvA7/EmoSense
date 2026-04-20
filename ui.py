"""
ui.py

Modern Tkinter UI for EmoSense:
- Clean layout and dark theme
- Chat bubbles simulation (user right, bot left)
- Status bar
- Dashboard & PDF Report integration
"""

import tkinter as tk
from tkinter import messagebox, scrolledtext
from tkinter import font as tkfont
import os

from auth import AuthSystem
from chatbot_engine import ChatbotEngine
from analysis import EmotionAnalyzer
from report_generator import generate_report


# --- Color Palette ---
BG_COLOR = "#1B1B2F"
PANEL_COLOR = "#22223B"
TEXT_COLOR = "#FFFFFF"
MUTED_TEXT = "#CCCCCC"
PRIMARY_COLOR = "#2E86AB"
ACCENT_COLOR = "#E27D60"
BOT_MSG_COLOR = "#41B3A3"
USER_MSG_COLOR = "#E8A87C"


class EmoSenseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EmoSense: Emotion-Aware Chatbot")
        self.root.geometry("850x650")
        self.root.configure(bg=BG_COLOR)
        self.root.minsize(700, 500)

        # Systems
        self.auth = AuthSystem()
        self.engine = ChatbotEngine()
        self.analyzer = EmotionAnalyzer()

        # Fonts
        self.title_font = tkfont.Font(family="Helvetica", size=28, weight="bold")
        self.header_font = tkfont.Font(family="Helvetica", size=16, weight="bold")
        self.normal_font = tkfont.Font(family="Helvetica", size=12)
        self.bold_font = tkfont.Font(family="Helvetica", size=12, weight="bold")
        self.small_font = tkfont.Font(family="Helvetica", size=10)

        self.current_frame = None
        self.show_login()

    def clear_screen(self):
        if self.current_frame:
            self.current_frame.destroy()
        self.current_frame = tk.Frame(self.root, bg=BG_COLOR)
        self.current_frame.pack(fill=tk.BOTH, expand=True)

    # ================================
    # LOGIN / SIGNUP SCREEN
    # ================================
    def show_login(self):
        self.clear_screen()

        # Center Frame
        center_frame = tk.Frame(self.current_frame, bg=PANEL_COLOR, padx=40, pady=40)
        center_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Title
        tk.Label(center_frame, text="EmoSense", font=self.title_font, fg=PRIMARY_COLOR, bg=PANEL_COLOR).pack(pady=(0, 10))
        tk.Label(center_frame, text="Log in or sign up to continue", font=self.normal_font, fg=MUTED_TEXT, bg=PANEL_COLOR).pack(pady=(0, 20))

        # Username
        tk.Label(center_frame, text="Username", font=self.bold_font, fg=TEXT_COLOR, bg=PANEL_COLOR).pack(anchor="w")
        self.username_entry = tk.Entry(center_frame, font=self.normal_font, width=30, bg="#4A4E69", fg=TEXT_COLOR, insertbackground=TEXT_COLOR, relief=tk.FLAT)
        self.username_entry.pack(pady=(5, 15), ipady=5)

        # Password
        tk.Label(center_frame, text="Password", font=self.bold_font, fg=TEXT_COLOR, bg=PANEL_COLOR).pack(anchor="w")
        self.password_entry = tk.Entry(center_frame, font=self.normal_font, width=30, show="*", bg="#4A4E69", fg=TEXT_COLOR, insertbackground=TEXT_COLOR, relief=tk.FLAT)
        self.password_entry.pack(pady=(5, 25), ipady=5)

        # Buttons
        btn_frame = tk.Frame(center_frame, bg=PANEL_COLOR)
        btn_frame.pack(fill=tk.X)

        login_btn = tk.Button(btn_frame, text="Login", font=self.bold_font, bg=PRIMARY_COLOR, fg=TEXT_COLOR, activebackground="#1f5d78", activeforeground=TEXT_COLOR, relief=tk.FLAT, cursor="hand2", command=self.login)
        login_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5), ipady=5)

        signup_btn = tk.Button(btn_frame, text="Signup", font=self.bold_font, bg=ACCENT_COLOR, fg=TEXT_COLOR, activebackground="#ba654d", activeforeground=TEXT_COLOR, relief=tk.FLAT, cursor="hand2", command=self.signup)
        signup_btn.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5, 0), ipady=5)

        # Bind Enter key
        self.root.bind('<Return>', lambda event: self.login())

    def login(self):
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()

        result = self.auth.login(username, password)
        if result["status"] == "success":
            self.root.unbind('<Return>')
            self.show_chat()
        else:
            messagebox.showerror("Login Error", result["message"])

    def signup(self):
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()

        result = self.auth.signup(username, password)
        if result["status"] == "success":
            messagebox.showinfo("Signup Success", "Account created successfully! You can now log in.")
        else:
            messagebox.showerror("Signup Error", result["message"])

    # ================================
    # CHAT SCREEN
    # ================================
    def show_chat(self):
        self.clear_screen()

        # 1. HEADER
        header_frame = tk.Frame(self.current_frame, bg=PANEL_COLOR, height=60, padx=20)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        tk.Label(header_frame, text="EmoSense", font=self.header_font, fg=PRIMARY_COLOR, bg=PANEL_COLOR).pack(side=tk.LEFT, pady=15)

        # Header Buttons
        btn_style = {"font": self.small_font, "fg": TEXT_COLOR, "bg": "#4A4E69", "activebackground": "#686D87", "activeforeground": TEXT_COLOR, "relief": tk.FLAT, "cursor": "hand2", "padx": 10, "pady": 5}
        
        tk.Button(header_frame, text="Logout", command=self.logout, **btn_style).pack(side=tk.RIGHT, pady=15, padx=(5, 0))
        tk.Button(header_frame, text="Clear Chat", command=self.clear_chat, **btn_style).pack(side=tk.RIGHT, pady=15, padx=5)
        tk.Button(header_frame, text="Generate Report", command=self.generate_report, **btn_style).pack(side=tk.RIGHT, pady=15, padx=5)
        tk.Button(header_frame, text="Dashboard", command=self.show_dashboard, **btn_style).pack(side=tk.RIGHT, pady=15, padx=5)

        # 2. STATUS BAR (at the bottom)
        status_frame = tk.Frame(self.current_frame, bg=PANEL_COLOR, height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(status_frame, text="Status: Ready", font=self.small_font, fg=MUTED_TEXT, bg=PANEL_COLOR)
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)

        # 3. INPUT AREA
        input_frame = tk.Frame(self.current_frame, bg=BG_COLOR, padx=20, pady=15)
        input_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.input_field = tk.Entry(input_frame, font=self.normal_font, bg="#4A4E69", fg=TEXT_COLOR, insertbackground=TEXT_COLOR, relief=tk.FLAT)
        self.input_field.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, ipady=10, padx=(0, 10))
        self.input_field.bind("<Return>", lambda event: self.send_message())

        send_btn = tk.Button(input_frame, text="Send", font=self.bold_font, bg=PRIMARY_COLOR, fg=TEXT_COLOR, activebackground="#1f5d78", activeforeground=TEXT_COLOR, relief=tk.FLAT, cursor="hand2", command=self.send_message)
        send_btn.pack(side=tk.RIGHT, ipady=5, ipadx=15)

        # 4. CHAT AREA
        chat_container = tk.Frame(self.current_frame, bg=BG_COLOR, padx=20, pady=10)
        chat_container.pack(fill=tk.BOTH, expand=True)

        self.chat_area = scrolledtext.ScrolledText(chat_container, wrap=tk.WORD, bg=BG_COLOR, fg=TEXT_COLOR, font=self.normal_font, relief=tk.FLAT, borderwidth=0, state='disabled')
        self.chat_area.pack(fill=tk.BOTH, expand=True)

        # Configure chat tags for "bubbles"
        self.chat_area.tag_configure("user_name", justify="right", foreground=USER_MSG_COLOR, font=self.bold_font)
        self.chat_area.tag_configure("user_msg", justify="right", foreground=TEXT_COLOR, font=self.normal_font, lmargin1=100, lmargin2=100, spacing3=15)
        
        self.chat_area.tag_configure("bot_name", justify="left", foreground=BOT_MSG_COLOR, font=self.bold_font)
        self.chat_area.tag_configure("bot_msg", justify="left", foreground=TEXT_COLOR, font=self.normal_font, rmargin=100, spacing1=2)
        self.chat_area.tag_configure("bot_meta", justify="left", foreground=MUTED_TEXT, font=self.small_font, rmargin=100, spacing3=15)

        self.chat_area.tag_configure("system", justify="center", foreground=MUTED_TEXT, font=self.small_font, spacing1=10, spacing3=10)

        # Welcome message
        self.append_system_msg("Welcome to EmoSense! Start chatting below.")
        self.input_field.focus_set()

    # ================================
    # CHAT FUNCTIONS
    # ================================
    def send_message(self):
        user_input = self.input_field.get().strip()
        user_id = self.auth.get_current_user()

        if not user_input:
            return

        # Disable input temporarily
        self.input_field.config(state='disabled')
        self.status_label.config(text="Status: Processing...")
        self.root.update()

        # Display user message
        self.append_user_msg(user_input)
        self.input_field.config(state='normal')
        self.input_field.delete(0, tk.END)

        # Process input
        try:
            result = self.engine.process_input(user_id, user_input)
            bot_reply = result["response"]
            emotion = result["emotion"]
            intensity = result["intensity"]
            
            # Display bot message
            self.append_bot_msg(bot_reply, emotion, intensity)
            
            # Update status
            self.status_label.config(text=f"Status: Ready | Last emotion detected: {emotion.capitalize()} (Intensity: {intensity})")
        except Exception as e:
            self.append_system_msg(f"Error processing message: {e}")
            self.status_label.config(text="Status: Error")

        self.input_field.focus_set()

    def append_user_msg(self, message):
        self.chat_area.config(state='normal')
        self.chat_area.insert(tk.END, "You\n", "user_name")
        self.chat_area.insert(tk.END, f"{message}\n", "user_msg")
        self.chat_area.config(state='disabled')
        self.chat_area.yview(tk.END)

    def append_bot_msg(self, message, emotion, intensity):
        self.chat_area.config(state='normal')
        self.chat_area.insert(tk.END, "EmoSense\n", "bot_name")
        self.chat_area.insert(tk.END, f"{message}\n", "bot_msg")
        self.chat_area.insert(tk.END, f"[Emotion: {emotion} | Intensity: {intensity}]\n", "bot_meta")
        self.chat_area.config(state='disabled')
        self.chat_area.yview(tk.END)

    def append_system_msg(self, message):
        self.chat_area.config(state='normal')
        self.chat_area.insert(tk.END, f"— {message} —\n", "system")
        self.chat_area.config(state='disabled')
        self.chat_area.yview(tk.END)

    def clear_chat(self):
        self.chat_area.config(state='normal')
        self.chat_area.delete(1.0, tk.END)
        self.chat_area.config(state='disabled')
        self.append_system_msg("Chat cleared.")

    def logout(self):
        self.auth.logout()
        self.show_login()

    # ================================
    # DASHBOARD
    # ================================
    def show_dashboard(self):
        user_id = self.auth.get_current_user()
        
        try:
            result = self.analyzer.analyze_user(user_id)
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")
            return

        if "error" in result:
            messagebox.showerror("Dashboard Info", "No data available yet. Start chatting to generate insights.")
            return

        metrics = result.get("metrics", {})
        patterns = result.get("patterns", {})
        risk = result.get("risk_level", "Unknown")
        message = result.get("message", "No message.")

        # Format Dashboard Info
        info = f"""--- METRICS ---
Average Intensity: {metrics.get('avg_intensity', 'N/A')}
Stability Score: {metrics.get('stability_score', 'N/A')}

--- PATTERNS ---
Dominant Emotion: {patterns.get('dominant_emotion', 'N/A').capitalize()}
Top Keywords: {', '.join([kw[0] for kw in patterns.get('top_keywords', [])]) if patterns.get('top_keywords') else 'None'}

--- RISK ASSESSMENT ---
Level: {risk}
{message}
"""
        messagebox.showinfo("Emotion Dashboard", info)

        # Trigger plots
        try:
            df = self.analyzer.load_user_data(user_id)
            if df is not None and not df.empty:
                self.analyzer.plot_emotion_distribution(df)
                self.analyzer.plot_emotion_trend(df)
        except Exception as e:
            print(f"[WARN] Could not display plots: {e}")

    # ================================
    # REPORT GENERATOR
    # ================================
    def generate_report(self):
        user_id = self.auth.get_current_user()
        self.status_label.config(text="Status: Generating PDF Report...")
        self.root.update()

        try:
            # 1. First get the analysis result
            analysis_result = self.analyzer.analyze_user(user_id)
            
            if "error" in analysis_result:
                messagebox.showerror("Report Error", "No data available to generate report. Start chatting first.")
                self.status_label.config(text="Status: Ready")
                return
                
            # 2. Add intensity_timeline for the chart
            df = self.analyzer.load_user_data(user_id)
            if df is not None and not df.empty:
                df_sorted = df.sort_values("timestamp")
                analysis_result["intensity_timeline"] = df_sorted["intensity"].tolist()
                
            # 3. Call generate_report with both arguments
            report_path = generate_report(user_id, analysis_result)
            
            messagebox.showinfo("Success", f"Report saved successfully!\n\nPath: {report_path}")
            self.status_label.config(text="Status: Report Generated")
        except Exception as e:
            messagebox.showerror("Report Error", f"Failed to generate report: {e}")
            self.status_label.config(text="Status: Error")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmoSenseApp(root)
    root.mainloop()