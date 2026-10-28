
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.agent import run_agent

app = FastAPI(
    title="AI Task Automation Agent",
    description="API de notre agent IA",
    version="0.2.0",
)


class TaskRequest(BaseModel):
    task: str


@app.get("/")
def home():
    return {
        "message": "AI Task Automation Agent API is running"
    }


@app.post("/tasks")
def create_task(request: TaskRequest):
    try:
        answer = run_agent(request.task)

        return {
            "task": request.task,
            "status": "success",
            "answer": answer
        }

    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Impossible de traiter la tâche pour le moment : {error}"
        )
