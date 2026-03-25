"""
app/api/v1/endpoints/auth.py
─────────────────────────────
Authentication endpoints.
- POST /auth/token   — get JWT access token + refresh token
- POST /auth/refresh — exchange refresh token for new access token
- GET  /auth/me      — current user info
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError, jwt
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
# Passwords are bcrypt-hashed.  Regenerate with:
#   python -c "from passlib.context import CryptContext; print(CryptContext(['bcrypt']).hash('pw'))"
_DEMO_USERS: dict[str, tuple[str, list[str], str]] = {
    "admin@company.com":   (_pwd_ctx.hash("admin123"),   ["admin"],         "Engineering"),
    "user@company.com":    (_pwd_ctx.hash("user123"),    ["user"],          "General"),
    "support@company.com": (_pwd_ctx.hash("support123"), ["support_agent"], "Customer Success"),
    "sales@company.com":   (_pwd_ctx.hash("sales123"),   ["sales_rep"],     "Sales"),
    "analyst@company.com": (_pwd_ctx.hash("analyst123"), ["analyst"],       "Finance"),
}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    roles: list[str]
    refresh_token: str = ""


def _create_token(
    user_id: str,
    roles: list[str],
    expires_minutes: int,
    token_type: str = "access",
) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    payload = {
        "sub": user_id,
        "roles": roles,
        "type": token_type,
        "exp": int(exp.timestamp()),
        "iat": int(datetime.now(timezone.utc).timestamp()),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def _authenticate(email: str, password: str) -> tuple[str, list[str], str] | None:
    """Verify credentials with bcrypt. Returns (email, roles, dept) or None."""
    record = _DEMO_USERS.get(email.lower())
    if not record:
        return None
    hashed_pw, roles, department = record
    if not _pwd_ctx.verify(password, hashed_pw):
        return None
    return email, roles, department


@router.post("/token", response_model=TokenResponse, summary="Get access token")
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
) -> TokenResponse:
    """
    OAuth2 password flow. Returns JWT access token + refresh token.

    Demo credentials (local/staging only):
      admin@company.com / admin123  |  user@company.com / user123
      support@company.com / support123  |  sales@company.com / sales123
    """
    result = _authenticate(form.username, form.password)
    if result is None:
        logger.warning("auth.login.failed", email=form.username[:20])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id, roles, _department = result

    access_token = _create_token(
        user_id=user_id,
        roles=roles,
        expires_minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
        token_type="access",
    )
    refresh_token = _create_token(
        user_id=user_id,
        roles=roles,
        expires_minutes=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60,
        token_type="refresh",
    )

    logger.info("auth.login.success", email=user_id[:20], roles=roles)
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
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
async def refresh_token_endpoint(body: RefreshRequest) -> TokenResponse:
    """
    Exchange a refresh token for a new access token.
    Validates the 'type' claim — access tokens are rejected.
    """
    try:
        payload = jwt.decode(
            body.refresh_token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        ) from exc

    if payload.get("type", "access") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not a refresh token — use the refresh_token field from /token",
        )

    user_id: str = payload.get("sub", "")
    roles: list[str] = payload.get("roles", [])

    new_access = _create_token(
        user_id=user_id,
        roles=roles,
        expires_minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
        token_type="access",
    )
    logger.info("auth.token.refreshed", user=user_id[:20])
    return TokenResponse(
        access_token=new_access,
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        roles=roles,
    )
