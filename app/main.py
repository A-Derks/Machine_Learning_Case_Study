from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import httpx
import os

app = FastAPI(title="My Portfolio · FastAPI")

# Static & templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# ---------- Pages ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    me = {"name": "Alysa Derks", "tagline": "Student · Developer · Problem Solver"}
    return templates.TemplateResponse("index.html", {"request": request, "title": "Home", "me": me})

@app.get("/introduction", response_class=HTMLResponse)
async def introduction(request: Request):
    return templates.TemplateResponse("introduction.html", {"request": request, "title": "Introduction"})

@app.get("/tab1", response_class=HTMLResponse)
async def tab1(request: Request):
    return templates.TemplateResponse("tab1.html", {"request": request, "title": "Tab1"})

@app.get("/tab2", response_class=HTMLResponse)
async def tab2(request: Request):
    return templates.TemplateResponse("tab2.html", {"request": request, "title": "Tab2"})


