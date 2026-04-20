"""
auth.py

Handles:
- User signup
- User login
- Validation layer between UI and database

Uses:
- database.py
"""

from database import Database


# ================================
# AUTH CLASS
# ================================

class AuthSystem:
    def __init__(self):
        self.db = Database()
        self.current_user = None  # stores logged-in user_id

    # ================================
    # SIGNUP
    # ================================
    def signup(self, username, password):
        """
        Create a new user with validation
        """

        # Basic validation
        if not username or not password:
            return {"status": "error", "message": "Fields cannot be empty"}

        if len(password) < 4:
            return {"status": "error", "message": "Password too short"}

        result = self.db.create_user(username, password)

        return result

    # ================================
    # LOGIN
    # ================================
    def login(self, username, password):
        """
        Authenticate user
        """

        if not username or not password:
            return {"status": "error", "message": "Fields cannot be empty"}

        result = self.db.authenticate_user(username, password)

        if result["status"] == "success":
            self.current_user = result["user_id"]

        return result

    # ================================
    # GET CURRENT USER
    # ================================
    def get_current_user(self):
        return self.current_user

    # ================================
    # LOGOUT
    # ================================
    def logout(self):
        self.current_user = None
        return {"status": "success", "message": "Logged out"}
        

# ================================
# TEST BLOCK
# ================================

if __name__ == "__main__":
    auth = AuthSystem()

    print("\n--- SIGNUP ---")
    res1 = auth.signup("atharva", "1234")
    print(res1)

    print("\n--- LOGIN ---")
    res2 = auth.login("atharva", "1234")
    print(res2)

    print("\n--- CURRENT USER ---")
    print(auth.get_current_user())

    print("\n--- LOGOUT ---")
    print(auth.logout())