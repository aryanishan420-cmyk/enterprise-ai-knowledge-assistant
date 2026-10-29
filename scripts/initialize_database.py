"""Create the SQLite tables (and data directory) configured by DATABASE_URL.

Usage:  python scripts/initialize_database.py
The API also creates tables on startup; this script is for explicit/CI initialisation.
"""

from backend.config import get_settings
from backend.database.repository import Repository

if __name__ == "__main__":
    s = get_settings()
    Repository(s.database_url)
    print(f"Database ready: {s.database_url}")
