from fastapi import FastAPI

app = FastAPI()

songs = [
    {"id": 1, "name": "Hello"},
    {"id": 2, "name": "Someone Like You"}
]


"""
@app.get("/")

This is called a route decorator.

It means:

If someone sends:

GET /

Call this function.

"""
@app.get("/imran")
def home():
    return songs