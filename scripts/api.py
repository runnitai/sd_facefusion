import logging

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status
from starlette.requests import Request

logger = logging.getLogger(__name__)


def facefusion_api(_, app: FastAPI):
    logger.debug("Loading Dreambooth API Endpoints.")

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
        )

    @app.get("/facefusion/queue/{job_id}")
    async def get_queue(job_id: str):
        from facefusion.uis.components.job_queue import JOB_QUEUE
        for job in JOB_QUEUE:
            if job.id == job_id:
                return job.to_dict()
        return {"queue": "queue"}

    @app.post("/facefusion/queue/")
    async def post_queue():
        from facefusion.uis.components.job_queue import JOB_QUEUE
        return {"queue": [job.to_dict() for job in JOB_QUEUE]}


try:
    from modules.shared import cmd_opts

    if cmd_opts.api:
        import modules.script_callbacks as script_callbacks

        script_callbacks.on_app_started(facefusion_api)
        logger.debug("SD-Webui API layer loaded")
    else:
        logger.debug("API flag not enabled, skipping API layer. Please enable with --api")
except:
    logger.debug("Unable to import script callbacks.")
    pass
