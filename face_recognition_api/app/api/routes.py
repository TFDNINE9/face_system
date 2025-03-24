from fastapi import APIRouter
from . import customers, health, events, photos, processing, persons, auth, users

api_router = APIRouter()

# Include all route modules
api_router.include_router(customers.router)
api_router.include_router(health.router)
api_router.include_router(events.router)
api_router.include_router(photos.router)
api_router.include_router(processing.router)
api_router.include_router(persons.router)
api_router.include_router(auth.router)
api_router.include_router(users.router)

# Add more routers here as your application grows
# api_router.include_router(face_recognition.router)