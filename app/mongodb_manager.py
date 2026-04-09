import os
import hashlib
import binascii
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

class MongoDBManager:
    def __init__(self):
        self.uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
        self.db_name = "wildguard_db"
        self.client = None
        self.db = None
        self._connect()

    def _connect(self):
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            self.db = self.client[self.db_name]
            # Test connection
            self.client.server_info()
            print("Successfully connected to MongoDB.")
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            self.db = None

    def _hash_password(self, password, salt=None):
        if salt is None:
            salt = binascii.hexlify(os.urandom(16)).decode('ascii')
        
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('ascii'), 100000)
        return binascii.hexlify(pwd_hash).decode('ascii'), salt

    def register_user(self, username, password):
        if self.db is None:
            return False, "Database connection failed."
        
        users = self.db.users
        if users.find_one({"username": username}):
            return False, "Username already exists."
        
        pwd_hash, salt = self._hash_password(password)
        
        try:
            users.insert_one({
                "username": username,
                "password_hash": pwd_hash,
                "salt": salt,
                "created_at": os.times()[4] # Approximate timestamp
            })
            return True, "User registered successfully."
        except Exception as e:
            return False, f"Registration failed: {e}"

    def authenticate_user(self, username, password):
        if self.db is None:
            return False, "Database connection failed."
        
        users = self.db.users
        user = users.find_one({"username": username})
        
        if not user:
            return False, "Invalid username or password."
        
        stored_hash = user['password_hash']
        salt = user['salt']
        
        new_hash, _ = self._hash_password(password, salt)
        
        if new_hash == stored_hash:
            return True, "Authentication successful."
        else:
            return False, "Invalid username or password."

# Singleton instance
db_manager = MongoDBManager()
