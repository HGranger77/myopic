from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from database import init_db  # noqa: E402 - must run after load_dotenv()
from routers import health, stories, watchlist  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="myopic watchlist API", lifespan=lifespan)
app.include_router(health.router)
app.include_router(watchlist.router)
app.include_router(stories.router)
