from fastapi import FastAPI
from router.receving_file import receive_file_router

app = FastAPI()

app.include_router(receive_file_router)