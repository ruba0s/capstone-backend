from fastapi import FastAPI
from app.api.routes import predict

app = FastAPI(title="BioAge API")

app.include_router(predict.router, tags=["prediction"])

@app.get("/")
def root():
    return {"status": "BioAge API is running"}