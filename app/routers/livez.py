from fastapi import APIRouter

router = APIRouter()


@router.api_route("/livez", methods=["GET", "HEAD"])
async def check_livez():
    return {"status": "live"}
