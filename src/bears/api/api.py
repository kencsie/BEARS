import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bears.api.routes.documents import router as documents_router
from bears.api.routes.experiments import router as experiments_router
from bears.api.routes.query import router as query_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from bears.core.dependencies import preload_all
    logger.info("Server startup: preloading singletons...")
    await asyncio.to_thread(preload_all)
    logger.info("Server startup complete")
    yield


app = FastAPI(title="BEARS API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
app.include_router(experiments_router)
app.include_router(documents_router)
