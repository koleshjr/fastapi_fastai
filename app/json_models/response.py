from pydantic import BaseModel
#for validation 
class Response(BaseModel):
    prediction: str
    probability: float