#!/usr/bin/env python3
"""
scripts/smoke_test.py
──────────────────────
Quick smoke test against a running API instance.
Run after 'make up' to verify everything is wired correctly.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --base-url http://staging.api.internal:8000
"""
import argparse
import sys

import httpx


def run(base_url: str) -> bool:
    passed = 0
    failed = 0

    def check(name: str, resp: httpx.Response, expected_status: int = 200) -> bool:
        nonlocal passed, failed
        ok = resp.status_code == expected_status
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} {name}: {resp.status_code}")
        if ok:
            passed += 1
        else:
            failed += 1
            print(f"    Response: {resp.text[:200]}")
        return ok

    print(f"\nSmoke test: {base_url}\n")

    with httpx.Client(base_url=base_url, timeout=10) as client:
        # 1. Health checks
        print("[1] Health checks")
        check("GET /health", client.get("/health"))
        check("GET /health/ready", client.get("/health/ready"))

        # 2. Auth
        print("\n[2] Auth")
        token_resp = client.post(
            "/api/v1/auth/token",
            data={"username": "user@company.com", "password": "user123"},
        )
        check("POST /auth/token (user)", token_resp)
        user_token = token_resp.json().get("access_token", "") if token_resp.status_code == 200 else ""

        admin_token_resp = client.post(
            "/api/v1/auth/token",
            data={"username": "admin@company.com", "password": "admin123"},
        )
        check("POST /auth/token (admin)", admin_token_resp)
        admin_token = admin_token_resp.json().get("access_token", "") if admin_token_resp.status_code == 200 else ""

        if user_token:
            check("GET /auth/me", client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {user_token}"},
            ))

        # 3. Query endpoint
        print("\n[3] Query")
        if user_token:
            check("POST /query (no auth) → 401", client.post(
                "/api/v1/query",
                json={"query": "test"},
            ), expected_status=401)

            check("POST /query (authed)", client.post(
                "/api/v1/query",
                json={"query": "What is the company vacation policy?"},
                headers={"Authorization": f"Bearer {user_token}"},
            ))

        # 4. Admin
        print("\n[4] Admin")
        if admin_token:
            check("GET /admin/plugins", client.get(
                "/admin/plugins",
                headers={"Authorization": f"Bearer {admin_token}"},
            ))

        # 5. RBAC check
        print("\n[5] RBAC")
        if user_token:
            check("GET /admin/plugins (user) → 403", client.get(
                "/admin/plugins",
                headers={"Authorization": f"Bearer {user_token}"},
            ), expected_status=403)

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    success = run(args.base_url)
    sys.exit(0 if success else 1)
