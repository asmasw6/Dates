
import os
import uvicorn
from main import app  # your FastAPI app

port = int(os.environ.get("PORT", 8000))  # Render injects this
uvicorn.run(app, host="0.0.0.0", port=port)
