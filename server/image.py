import mimetypes

from fastapi import UploadFile

from .models import AppError, PreparedImage, RequestLog


class ImagePreparer:
    @staticmethod
    def from_upload(upload: UploadFile, *, max_image_bytes: int, log: RequestLog) -> PreparedImage:
        if not upload.filename:
            raise AppError(400, "Image filename is missing.")

        log.add(f"Received upload: {upload.filename}")

        if upload.content_type and not upload.content_type.startswith("image/"):
            raise AppError(400, "Only image uploads are allowed.")

        raw_bytes = upload.file.read()
        if not raw_bytes:
            raise AppError(400, "Uploaded image is empty.")

        log.add(f"Read upload bytes: {len(raw_bytes)}")

        if len(raw_bytes) > max_image_bytes:
            raise AppError(400, "Image exceeds the 15 MB limit.")

        prepared = PreparedImage(
            bytes_data=raw_bytes,
            mime_type=ImagePreparer._resolve_mime_type(upload),
        )

        log.add(
            "Using original upload bytes for vLLM. "
            f"mime: {prepared.mime_type}, "
            f"bytes: {len(prepared.bytes_data)}"
        )
        log.add("Encoded original upload as data URL for vLLM")
        return prepared

    @staticmethod
    def _resolve_mime_type(upload: UploadFile) -> str:
        if upload.content_type:
            return upload.content_type

        guessed_type, _ = mimetypes.guess_type(upload.filename or "")
        if guessed_type:
            return guessed_type

        return "image/jpeg"

