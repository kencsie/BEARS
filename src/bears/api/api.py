from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from bears.api.routes.evaluation import router as eval_router
from bears.api.routes.query import router as query_router
from bears.api.routes.experiments import router as experiments_router

app = FastAPI(title="BEARS API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(eval_router)
app.include_router(query_router)
app.include_router(experiments_router)
