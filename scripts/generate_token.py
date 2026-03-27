#!/usr/bin/env python3
"""
scripts/generate_token.py
──────────────────────────
Dev utility: generate a JWT token for API testing.

Usage:
    python scripts/generate_token.py --email admin@company.com --roles admin
    python scripts/generate_token.py --email sales@company.com --roles sales_rep user
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("ENVIRONMENT", "local")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JWT token for dev/testing")
    parser.add_argument("--email", default="admin@company.com")
    parser.add_argument("--roles", nargs="+", default=["admin"])
    parser.add_argument("--expires-hours", type=int, default=24)
    args = parser.parse_args()

    # Load settings after env is set
    from app.core.config import settings
    from jose import jwt

    payload = {
        "sub": args.email,
        "roles": args.roles,
        "exp": int((datetime.utcnow() + timedelta(hours=args.expires_hours)).timestamp()),
        "iat": int(datetime.utcnow().timestamp()),
    }
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    print(f"\nToken for: {args.email}")
    print(f"Roles: {args.roles}")
    print(f"Expires: {args.expires_hours}h\n")
    print(f"Bearer {token}\n")
    print("Usage:")
    print(f'  curl -H "Authorization: Bearer {token}" http://localhost:8000/api/v1/query ...')


if __name__ == "__main__":
    main()
