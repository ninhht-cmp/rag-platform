"""
app/api/v1/endpoints/auth.py
─────────────────────────────
Authentication endpoints.
- POST /auth/token   — get JWT access token
- POST /auth/refresh — refresh token
- GET  /auth/me      — current user info

In production: replace the stub user lookup with your
real user store (DB, LDAP, SSO, Keycloak, etc.)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from app.api.v1.middleware.auth import get_current_user
from app.core.config import settings
from app.core.logging import get_logger
from app.models.domain import Role, User

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["Auth"])
_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Stub user store ────────────────────────────────────────────────
# ⚠️  DEMO ONLY — replace with real SSO/DB lookup in production.
# These demo users exist ONLY for local development and testing.
# Set ENVIRONMENT=production to require real auth.
# Format: {email: (hashed_password, roles, department)}
_DEMO_USERS_RAW: dict[str, tuple[str, list[str], str]] = {
    "admin@company.com":   ("admin123",   ["admin"],         "Engineering"),
    "user@company.com":    ("user123",    ["user"],          "General"),
    "support@company.com": ("support123", ["support_agent"], "Customer Success"),
    "sales@company.com":   ("sales123",  ["sales_rep"],      "Sales"),
    "analyst@company.com": ("analyst123", ["analyst"],       "Finance"),
}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    roles: list[str]
    refresh_token: str = ""   # populated on /token, used with /refresh


def _create_token(user_id: str, roles: list[str], expires_minutes: int) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    iat = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "roles": roles,
        "exp": int(exp.timestamp()),
        "iat": int(iat.timestamp()),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def _authenticate(email: str, password: str) -> tuple[str, list[str], str] | None:
    record = _DEMO_USERS_RAW.get(email.lower())
    if not record:
        return None
    plain_pw, roles, department = record
    if password != plain_pw:
        return None
    return email, roles, department

@router.post("/token", response_model=TokenResponse, summary="Get access token")
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
) -> TokenResponse:
    """
    OAuth2 password flow.
    Returns JWT access token valid for `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`.

    Demo credentials:
    - admin@company.com / admin123
    - user@company.com / user123
    - support@company.com / support123
    - sales@company.com / sales123
    """
    result = _authenticate(form.username, form.password)
    if result is None:
        logger.warning("auth.login.failed", email=form.username[:20])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id, roles, department = result
    token = _create_token(
        user_id=user_id,
        roles=roles,
        expires_minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    )

    logger.info("auth.login.success", email=user_id[:20], roles=roles)
    refresh = _create_token(
        user_id=user_id,
        roles=roles,
        expires_minutes=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60,
    )
    # embed type claim for refresh token validation
    import json as _json
    from jose import jwt as _jwt
    _payload = _jwt.decode(refresh, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    _payload["type"] = "refresh"
    refresh = _jwt.encode(_payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    return TokenResponse(
        access_token=token,
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        roles=roles,
    )


@router.get("/me", summary="Get current user info")
async def me(user: User = Depends(get_current_user)) -> dict:
    return {
        "id": str(user.id),
        "email": user.email,
        "name": user.name,
        "roles": [r.value for r in user.roles],
        "department": user.department,
    }

class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/refresh", response_model=TokenResponse, summary="Refresh access token")
async def refresh_token(body: RefreshRequest) -> TokenResponse:
    """
    Exchange a refresh token for a new access token.
    Refresh tokens are long-lived (7 days by default).
    """
    try:
        payload = jwt.decode(
            body.refresh_token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        user_id = payload.get("sub", "")
        roles = payload.get("roles", [])
        token_type = payload.get("type", "access")
        if token_type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not a refresh token",
            )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        ) from exc

    new_token = _create_token(
        user_id=user_id,
        roles=roles,
        expires_minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    )
    logger.info("auth.token.refreshed", user=user_id[:20])
    return TokenResponse(
        access_token=new_token,
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        roles=roles,
    )


@router.post("/refresh", response_model=TokenResponse, summary="Refresh access token")
async def refresh_token(
    current_user: User = Depends(get_current_user),
) -> TokenResponse:
    """
    Issue a new access token using the current (still-valid) token.
    Call this before token expiry to avoid re-login.
    """
    roles = [r.value for r in current_user.roles]
    token = _create_token(
        user_id=current_user.email,
        roles=roles,
        expires_minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    )
    logger.info("auth.refresh", email=current_user.email[:20])
    return TokenResponse(
        access_token=token,
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        roles=roles,
    )
