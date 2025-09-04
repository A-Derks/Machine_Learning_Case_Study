# FastAPI Portfolio with Tabs

Routes:
- `/` (Home)
- `/introduction`
- `/tab1`
- `/tab2`

API:
- `GET /api/github_repos?user=<username>&limit=5`

## Local Run
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://127.0.0.1:8000

## Deploy (Render)
- Build: `pip install -r requirements.txt`
- Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
