import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import ResponseValidationError

from app.api.v1.routes import router as v1_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cybersecurity Copilot PoC", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.exception_handler(ResponseValidationError)
async def response_validation_exception_handler(
    request: Request,
    exc: ResponseValidationError,
):
    logger.exception("Response validation failed on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Response validation failed (LLM output did not match schema).",
        },
    )


app.include_router(v1_router, prefix="/v1")
