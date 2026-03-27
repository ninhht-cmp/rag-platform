"""
app/api/v1/endpoints/ingestion.py
──────────────────────────────────
Document ingestion endpoints.
- POST /ingest/upload   — upload file for indexing
- DELETE /ingest/{id}   — remove document from index
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.v1.middleware.auth import get_current_user, require_roles
from app.core.logging import get_logger
from app.models.domain import Document, DocumentStatus, IngestionResult, Role, User
from app.services.ingestion.ingestion_service import get_ingestion_service

logger = get_logger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
    "text/html",
}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@router.post(
    "/upload",
    response_model=IngestionResult,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and index a document",
)
async def upload_document(
    use_case_id: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
    user: User = Depends(get_current_user),  # noqa: B008
) -> IngestionResult:
    """
    Upload a document for ingestion into the specified use case.

    Supported formats: PDF, DOCX, TXT, MD, HTML
    Max size: 50 MB

    The document will be:
    1. Extracted to plain text
    2. PII redacted (per plugin config)
    3. Chunked and embedded
    4. Indexed in Qdrant
    """
    # Validate content type
    content_type = file.content_type or "application/octet-stream"
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {content_type}. Allowed: {ALLOWED_CONTENT_TYPES}",
        )

    # Read and validate size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {len(content):,} bytes. Max: {MAX_FILE_SIZE:,} bytes",
        )

    document = Document(
        id=str(uuid.uuid4()),
        use_case_id=use_case_id,
        user_id=str(user.id),
        filename=file.filename or "unknown",
        content_type=content_type,
        size_bytes=len(content),
        status=DocumentStatus.PROCESSING,
        metadata={
            "uploaded_by": str(user.id),
            "department": user.department,
        },
    )

    svc = get_ingestion_service()
    result = await svc.ingest_document(
        document=document,
        content=content,
        use_case_id=use_case_id,
    )

    if result.status == DocumentStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result.error_message or "Ingestion failed",
        )

    logger.info(
        "api.ingest.upload",
        doc_id=document.id,
        use_case=use_case_id,
        chunks=result.chunks_created,
        user=str(user.id),
    )
    return result


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a document from the index",
)
async def delete_document(
    document_id: str,
    use_case_id: str,
    user: User = Depends(require_roles(Role.ADMIN, Role.ANALYST)),  # noqa: B008
) -> None:
    """Remove all chunks of a document from the vector index."""
    svc = get_ingestion_service()
    await svc.delete_document(document_id=document_id, use_case_id=use_case_id)
    logger.info(
        "api.ingest.delete",
        doc_id=document_id,
        use_case=use_case_id,
        user=str(user.id),
    )


@router.get(
    "/documents",
    summary="List documents uploaded by current user",
)
async def list_documents(
    use_case_id: str | None = None,
    user: User = Depends(get_current_user),  # noqa: B008
) -> dict:
    """
    List documents the current user has uploaded.
    Optionally filter by use_case_id.
    """
    # When DocumentRepository is connected to DB, replace with real query.
    # For now returns a stub — connect to DocumentRepository.get_by_user()
    return {
        "user_id": str(user.id),
        "use_case_id": use_case_id,
        "documents": [],
        "note": "Connect DocumentRepository to return real data",
    }
