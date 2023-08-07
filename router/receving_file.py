from fastapi import APIRouter, HTTPException, UploadFile, File

receive_file_router = APIRouter()

@receive_file_router.post("/receive_file")
async def receive_file(wave_file: UploadFile = File(None)):
    if wave_file is None:
        raise HTTPException(status_code=400, detail="파일이 정상적으로 전달되지 않았습니다.")
    else:
        with open(wave_file.filename, "wb") as f:
            f.write(wave_file.file.read())
        return { "filename" : wave_file.filename }