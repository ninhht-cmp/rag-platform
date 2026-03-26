"""
app/api/v1/middleware/auth.py
──────────────────────────────
JWT authentication middleware.
- Bearer token validation
- Role extraction
- User context injection via request.state
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.core.config import settings
from app.core.logging import get_logger
from app.models.domain import Role, TokenPayload, User

logger = get_logger(__name__)
_bearer = HTTPBearer(auto_error=False)


def _decode_token(token: str) -> TokenPayload:
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return TokenPayload(**payload)
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),  # noqa: B008
) -> User:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    payload = _decode_token(credentials.credentials)
    import uuid as _uuid
    try:
        user_id = _uuid.UUID(payload.sub)
    except (ValueError, AttributeError):
        # sub is email or arbitrary string — generate stable UUID from it
        user_id = _uuid.uuid5(_uuid.NAMESPACE_URL, payload.sub)
    return User(
        id=user_id,
        email=payload.sub if "@" in payload.sub else f"{payload.sub}@platform.local",
        name=payload.sub,
        roles=[Role(r) for r in payload.roles if r in {m.value for m in Role}],
    )


def require_roles(*roles: Role) -> User:
    """Dependency factory: check user has at least one of the required roles."""
    async def _check(user: User = Depends(get_current_user)) -> User:  # noqa: B008
        if not any(r in user.roles for r in roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required roles: {[r.value for r in roles]}",
            )
        return user
    return _check  # type: ignore[return-value]
