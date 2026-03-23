"""
tests/unit/test_auth.py
────────────────────────
Unit tests for JWT auth middleware and token generation.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt

from app.api.v1.middleware.auth import _decode_token, get_current_user
from app.core.config import settings
from app.models.domain import Role


def make_token(
    sub: str = "user@co.com",
    roles: list[str] | None = None,
    exp_delta_minutes: int = 60,
) -> str:
    payload = {
        "sub": sub,
        "roles": roles or ["user"],
        "exp": int((datetime.utcnow() + timedelta(minutes=exp_delta_minutes)).timestamp()),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


class TestJWTDecode:
    def test_valid_token_decodes_correctly(self) -> None:
        token = make_token(sub="test@co.com", roles=["admin", "user"])
        payload = _decode_token(token)
        assert payload.sub == "test@co.com"
        assert "admin" in payload.roles

    def test_expired_token_raises_401(self) -> None:
        token = make_token(exp_delta_minutes=-5)  # expired 5 minutes ago
        with pytest.raises(HTTPException) as exc_info:
            _decode_token(token)
        assert exc_info.value.status_code == 401

    def test_invalid_token_raises_401(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            _decode_token("totally.invalid.token")
        assert exc_info.value.status_code == 401

    def test_wrong_secret_raises_401(self) -> None:
        payload = {
            "sub": "user@co.com",
            "roles": ["user"],
            "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
        }
        token = jwt.encode(payload, "WRONG_SECRET", algorithm="HS256")
        with pytest.raises(HTTPException) as exc_info:
            _decode_token(token)
        assert exc_info.value.status_code == 401

    def test_missing_sub_field_raises_401(self) -> None:
        payload = {
            "roles": ["user"],
            "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
        }
        token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
        # TokenPayload requires sub — should fail validation
        with pytest.raises(Exception):
            payload_obj = _decode_token(token)


class TestGetCurrentUser:
    @pytest.mark.asyncio
    async def test_valid_credentials_returns_user(self) -> None:
        token = make_token(sub="alice@co.com", roles=["admin", "user"])
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        user = await get_current_user(creds)
        assert user.email == "alice@co.com"
        assert Role.ADMIN in user.roles
        assert Role.USER in user.roles

    @pytest.mark.asyncio
    async def test_none_credentials_raises_401(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(None)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_unknown_role_is_filtered(self) -> None:
        """Roles not in the Role enum are silently dropped."""
        payload = {
            "sub": "user@co.com",
            "roles": ["user", "super_secret_role_that_does_not_exist"],
            "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
        }
        token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        user = await get_current_user(creds)
        assert Role.USER in user.roles
        assert len(user.roles) == 1  # invalid role dropped


class TestRequireRoles:
    @pytest.mark.asyncio
    async def test_user_with_required_role_passes(self) -> None:
        from app.api.v1.middleware.auth import require_roles
        from app.models.domain import User
        import uuid

        admin_user = User(
            id=uuid.uuid4(), email="a@co.com", name="Admin",
            roles=[Role.ADMIN]
        )
        checker = require_roles(Role.ADMIN)
        # Should not raise
        result = await checker(admin_user)
        assert result is admin_user

    @pytest.mark.asyncio
    async def test_user_without_required_role_raises_403(self) -> None:
        from app.api.v1.middleware.auth import require_roles
        from app.models.domain import User
        import uuid

        regular_user = User(
            id=uuid.uuid4(), email="u@co.com", name="User",
            roles=[Role.USER]
        )
        checker = require_roles(Role.ADMIN)
        with pytest.raises(HTTPException) as exc_info:
            await checker(regular_user)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_any_matching_role_passes(self) -> None:
        from app.api.v1.middleware.auth import require_roles
        from app.models.domain import User
        import uuid

        sales_user = User(
            id=uuid.uuid4(), email="s@co.com", name="Sales",
            roles=[Role.SALES_REP]
        )
        # Require admin OR sales_rep — sales_rep should pass
        checker = require_roles(Role.ADMIN, Role.SALES_REP)
        result = await checker(sales_user)
        assert result is sales_user
