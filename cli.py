#!/usr/bin/env python3
"""
CLI tool for managing user tokens and other operations.
"""

import argparse
import sys
from pathlib import Path
import google.auth.transport.requests
import google.oauth2.credentials

# Add the current directory to sys.path to import from main.py
sys.path.insert(0, str(Path(__file__).parent))

from main import (
    load_user_tokens,
    save_user_tokens,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET
)

def refresh_user_token(uid: str) -> bool:
    """
    Refresh the OAuth token for a given user ID.

    Args:
        uid: User ID

    Returns:
        True if refresh was successful, False otherwise
    """
    try:
        # Load current credentials
        creds = load_user_tokens(uid)
        if not creds:
            print(f"No tokens found for user {uid}")
            return False

        # Check if token needs refresh
        if creds.expired and creds.refresh_token:
            print(f"Token expired for user {uid}, refreshing...")
            creds.refresh(google.auth.transport.requests.Request())
            print("Token refreshed successfully")

            # Save updated credentials
            save_user_tokens(uid, creds)
            return True
        elif creds.expired:
            print(f"Token expired for user {uid} but no refresh token available")
            return False
        else:
            print(f"Token for user {uid} is still valid")
            return True

    except Exception as e:
        print(f"Error refreshing token for user {uid}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="CLI tool for omi-notes")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # refresh-token command
    refresh_parser = subparsers.add_parser('refresh-token', help='Refresh OAuth token for a user')
    refresh_parser.add_argument('uid', help='User ID to refresh token for')

    args = parser.parse_args()

    if args.command == 'refresh-token':
        success = refresh_user_token(args.uid)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()