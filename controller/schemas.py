from pydantic import BaseModel


class HealthResponse(BaseModel):
    controller: str
