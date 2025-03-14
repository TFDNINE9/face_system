from fastapi.responses import JSONResponse
from fastapi import status

def create_response(body: dict, status_code: int = status.HTTP_200_OK, headers=None):
    return JSONResponse(
        content=body,
        status_code=status_code,
        headers=headers
    )