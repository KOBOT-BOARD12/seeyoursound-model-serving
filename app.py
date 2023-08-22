from fastapi import FastAPI
from router.receving_file import receive_file_router
from router.stt_serving import stt_serving_router

app = FastAPI()

app.include_router(receive_file_router)
app.include_router(stt_serving_router)