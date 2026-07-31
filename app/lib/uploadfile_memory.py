import io


class UploadFileInMemory:
    def __init__(self, file_content: bytes, filename: str, content_type: str):
        self.file_content = file_content
        self.filename = filename
        self.file = io.BytesIO(file_content)
        self.content_type = content_type
        self.size = len(file_content)

    async def read(self, size: int = -1) -> bytes:
        return self.file.read(size)

    async def seek(self, offset: int) -> None:
        self.file.seek(offset)

    async def close(self) -> None:
        self.file.close()
