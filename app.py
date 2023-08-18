from fastapi import FastAPI
from router.receving_file import receive_file_router
from router.classficating_sound import classficating_sound_router
app = FastAPI()

app.include_router(receive_file_router)
app.include_router(classficating_sound_router)