import hashlib
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str


def _compute_token(username: str, password: str) -> str:
    secret = os.environ.get("APP_SECRET_TOKEN", "ai-hedge-fund-default-salt")
    return hashlib.sha256(f"{username}:{password}:{secret}".encode()).hexdigest()


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    expected_username = os.environ.get("APP_USERNAME", "")
    expected_password = os.environ.get("APP_PASSWORD", "")

    if not expected_username or not expected_password:
        raise HTTPException(status_code=500, detail="Auth not configured on server")

    if request.username != expected_username or request.password != expected_password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return LoginResponse(token=_compute_token(expected_username, expected_password))
