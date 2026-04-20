"""
database.py

Handles:
- MongoDB connection
- User operations
- Chat storage
- Data retrieval

Library:
- pymongo
"""

from pymongo import MongoClient
from datetime import datetime
import uuid


# ================================
# DATABASE CLASS
# ================================

class Database:
    def __init__(self):
        """
        Initialize MongoDB connection
        """

        # Connect to MongoDB (with a short timeout so the app doesn't hang)
        self.client = MongoClient(
            "mongodb://localhost:27017/",
            serverSelectionTimeoutMS=3000,
        )

        # Create / connect database
        self.db = self.client["emosense"]

        # Collections
        self.users = self.db["users"]
        self.chats = self.db["chats"]

        # Verify the connection is live
        try:
            self.client.admin.command("ping")
            self.connected = True
            print("[OK] Connected to MongoDB (emosense)")
        except Exception as e:
            self.connected = False
            print(f"[WARN] MongoDB not reachable: {e}")
            print("   The app will run, but data will NOT be saved.")

    # ================================
    # USER FUNCTIONS
    # ================================

    def create_user(self, username, password):
        """
        Create new user
        """
        try:
            # Check if username exists
            if self.users.find_one({"username": username}):
                return {"status": "error", "message": "Username already exists"}

            user_id = str(uuid.uuid4())

            self.users.insert_one({
                "user_id": user_id,
                "username": username,
                "password": password  # (we'll hash later if needed)
            })

            return {"status": "success", "user_id": user_id}
        except Exception as e:
            return {"status": "error", "message": f"Database error: {e}"}

    def authenticate_user(self, username, password):
        """
        Login user
        """
        try:
            user = self.users.find_one({
                "username": username,
                "password": password
            })

            if user:
                return {"status": "success", "user_id": user["user_id"]}
            else:
                return {"status": "error", "message": "Invalid credentials"}
        except Exception as e:
            return {"status": "error", "message": f"Database error: {e}"}

    # ================================
    # CHAT FUNCTIONS
    # ================================

    def save_chat(self, user_id, message, emotion, intensity, context):
        """
        Save chat entry
        """
        try:
            self.chats.insert_one({
                "user_id": user_id,
                "message": message,
                "emotion": emotion,
                "intensity": intensity,
                "context": context,
                "timestamp": datetime.now()
            })
        except Exception as e:
            print(f"[WARN] Could not save chat: {e}")

    def get_user_chats(self, user_id):
        """
        Retrieve all chats of a user
        """
        try:
            chats = list(self.chats.find({"user_id": user_id}))
            return chats
        except Exception as e:
            print(f"[WARN] Could not fetch chats: {e}")
            return []

    # ================================
    # ANALYSIS SUPPORT FUNCTIONS
    # ================================

    def get_recent_chats(self, user_id, limit=50):
        """
        Get recent chats (for trend analysis)
        """
        try:
            chats = list(
                self.chats.find({"user_id": user_id})
                .sort("timestamp", -1)
                .limit(limit)
            )

            return chats
        except Exception as e:
            print(f"[WARN] Could not fetch recent chats: {e}")
            return []

    def get_all_users(self):
        """
        Optional: admin usage
        """
        try:
            return list(self.users.find())
        except Exception as e:
            print(f"[WARN] Could not fetch users: {e}")
            return []


# ================================
# TEST BLOCK
# ================================

if __name__ == "__main__":
    db = Database()

    # Create test user
    result = db.create_user("test_user", "1234")
    print(result)

    # Login test
    login = db.authenticate_user("test_user", "1234")
    print(login)

    if login["status"] == "success":
        uid = login["user_id"]

        # Save chat
        db.save_chat(
            user_id=uid,
            message="I feel very sad",
            emotion="sad",
            intensity=0.8,
            context="general"
        )

        # Fetch chats
        chats = db.get_user_chats(uid)
        print(f"Total chats: {len(chats)}")