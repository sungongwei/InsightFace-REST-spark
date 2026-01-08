import logging
import os
import ssl
from contextlib import asynccontextmanager

import aiohttp
from aiohttp import ClientTimeout, TCPConnector
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline
from fastapi.staticfiles import StaticFiles

from if_rest.api.routes.v1 import v1_router
from if_rest.core.processing import get_processing
from if_rest.core.face_manager import FaceManager
from if_rest.api.routes.v1.face_management import set_face_manager
from if_rest.logger import logger
from if_rest.settings import Settings

import os

__version__ = os.getenv('IFR_VERSION', '0.9.5.0')

dir_path = os.path.dirname(os.path.realpath(__file__))

# Read runtime settings from environment variables
settings = Settings()

logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Perform any necessary setup when the application starts up.
    This includes initializing the `processing` object aiohttp.ClientSession.

    Raises:
        Exception: If an error occurs during processing initialization.
    """

    logger.info(f"Starting processing module...")
    try:
        timeout = ClientTimeout(total=60.)
        if settings.defaults.sslv3_hack:
            ssl_context = ssl._create_unverified_context()
            ssl_context.set_ciphers('DEFAULT')
            dl_client = aiohttp.ClientSession(timeout=timeout, connector=TCPConnector(ssl=ssl_context))
        else:
            dl_client = aiohttp.ClientSession(timeout=timeout, connector=TCPConnector(ssl=False))
        processing = await get_processing()
        await processing.start(dl_client=dl_client)
        logger.info(f"Processing module ready!")

        # Initialize face manager with GPU support based on environment variable
        use_gpu = os.getenv('USE_GPU', 'false').lower() == 'true'
        face_manager = FaceManager(processing=processing, use_gpu=use_gpu)
        set_face_manager(face_manager)
        logger.info(f"Face manager ready! GPU support: {use_gpu}")
    except Exception as e:
        logger.error(e)
        exit(1)
    yield


from fastapi.responses import HTMLResponse
from fastapi import Request


def get_app() -> FastAPI:
    application = FastAPIOffline(
        title="InsightFace-REST",
        description="Face recognition REST API",
        version=__version__,
        lifespan=lifespan
    )

    application.add_middleware(
        CORSMiddleware,  # noqa
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*']
    )
    application.include_router(v1_router)

    # Serve static files
    application.mount("/static", StaticFiles(directory="if_rest/static"), name="static")

    @application.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        with open("if_rest/static/index.html", "r") as f:
            content = f.read()
        return HTMLResponse(content=content)

    return application


app = get_app()
