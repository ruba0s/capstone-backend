from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import predict, explain, recommendations

app = FastAPI(title="BioAge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, tags=["prediction"])
app.include_router(explain.router, tags=["explanation"])
app.include_router(recommendations.router, tags=["recommendations"])

@app.get("/")
def root():
    return {"status": "BioAge API is running"}