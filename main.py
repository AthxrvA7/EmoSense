"""
main.py

Entry point of EmoSense application.

This file:
- Starts the UI
- Acts as central runner for the project
"""

import tkinter as tk
from ui import EmoSenseApp


def main():
    """
    Main function to launch application
    """

    print("Starting EmoSense Application...")

    root = tk.Tk()

    # Initialize app
    app = EmoSenseApp(root)

    # Start GUI loop
    root.mainloop()


# ================================
# RUN APP
# ================================
if __name__ == "__main__":
    main()