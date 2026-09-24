"""API package — registers all routers."""

from fastapi import APIRouter

from app.api.submissions import router as submissions_router
from app.api.images import router as images_router
from app.api.text import router as text_router
from app.api.search import router as search_router

api_router = APIRouter()
api_router.include_router(submissions_router, prefix="/submissions", tags=["submissions"])
api_router.include_router(images_router, prefix="/images", tags=["images"])
api_router.include_router(text_router, prefix="/text", tags=["text"])
api_router.include_router(search_router, prefix="/search", tags=["search"])
