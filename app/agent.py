
import os
import json
import time
from typing import Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, START, END

from app.tools.calculator import add, multiply
from app.tools.date_tool import get_current_date


# 1. Configuration de Gemini
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY est absente du fichier .env"
    )

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-3.5-flash"
MAX_TOOL_CALLS = 3


def generate_content_with_retry(**kwargs):
    """Réessaie les erreurs temporaires de Gemini."""
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(**kwargs)

        except Exception as error:
            message = str(error).lower()
            temporary_error = any(
                marker in message
                for marker in [
                    "503",
                    "429",
                    "unavailable",
                    "overloaded",
                    "resource_exhausted",
                    "deadline exceeded",
                ]
            )

            if not temporary_error or attempt == max_attempts - 1:
                raise

            delay = 2 ** attempt
            print(
                f"Erreur temporaire Gemini. "
                f"Nouvel essai dans {delay} seconde(s)..."
            )
            time.sleep(delay)


# 2. Définition des outils Python
def add_tool(a: float, b: float) -> dict:
    """Additionne deux nombres."""
    return {"result": add(a, b)}


def multiply_tool(a: float, b: float) -> dict:
    """Multiplie deux nombres."""
    return {"result": multiply(a, b)}

def get_current_date_tool() -> dict:
    return {
        "result": get_current_date()
    }


TOOLS = {
    "add": add_tool,
    "multiply": multiply_tool,
    "get_current_date": get_current_date_tool,
}



# 3. Déclaration des outils accessibles à Gemini
TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="add",
        description="Additionne deux nombres.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "a": {
                    "type": "NUMBER",
                    "description": "Premier nombre",
                },
                "b": {
                    "type": "NUMBER",
                    "description": "Deuxième nombre",
                },
            },
            "required": ["a", "b"],
        },
    ),
    types.FunctionDeclaration(
        name="multiply",
        description="Multiplie deux nombres.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "a": {
                    "type": "NUMBER",
                    "description": "Premier nombre",
                },
                "b": {
                    "type": "NUMBER",
                    "description": "Deuxième nombre",
                },
            },
            "required": ["a", "b"],
        },
    ),
    types.FunctionDeclaration(
        name="get_current_date",
        description="Retourne la date et l'heure actuelles.",
        parameters={
            "type": "OBJECT",
            "properties": {},
        },
    ),
]

GEMINI_TOOLS = [
    types.Tool(function_declarations=TOOL_DECLARATIONS)
]




# 4. État partagé entre les étapes de LangGraph
class AgentState(TypedDict):
    task: str
    answer: str
    tool_name: str
    tool_args: dict
    tool_result: str
    tool_history: list[str]
    last_tool_name: str
    steps: int


def ask_gemini(state: AgentState):
    contents = (
        f"Tâche de l'utilisateur : {state['task']}\n\n"
        "Tu es un assistant intelligent qui peut utiliser trois outils : "
        "add, multiply et get_current_date.\n\n"
        "Règles d'utilisation des outils :\n"
        "1. Utilise add pour additionner deux nombres.\n"
        "2. Utilise multiply pour multiplier deux nombres.\n"
        "3. Utilise get_current_date pour obtenir la date et "
        "l'heure actuelles.\n"
        "4. Si plusieurs opérations sont nécessaires, "
        "utilise les résultats précédents.\n"
        "5. N'invente jamais les résultats des outils.\n\n"
        "Quand tu as terminé, donne une réponse finale claire "
        "en français."
    )


    if state["tool_history"]:
        contents += (
            "\n\nHistorique des outils exécutés :\n"
            + "\n".join(state["tool_history"])
        )

    if state["steps"] >= MAX_TOOL_CALLS:
        contents += (
            "\n\nLa limite d'appels d'outils est atteinte. "
            "N'appelle aucun outil supplémentaire. "
            "Donne la réponse finale à partir des résultats disponibles."
        )

    response = generate_content_with_retry(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            tools=GEMINI_TOOLS,
            automatic_function_calling=(
                types.AutomaticFunctionCallingConfig(
                    disable=True
                )
            ),
        ),
    )

    if not response.candidates:
        return {
            "answer": "Gemini n'a pas fourni de réponse.",
            "tool_name": "",
            "tool_args": {},
        }

    parts = response.candidates[0].content.parts

    for part in parts:
        if part.function_call and state["steps"] < MAX_TOOL_CALLS:
            function_call = part.function_call

            return {
                "tool_name": function_call.name,
                "tool_args": dict(function_call.args or {}),
            }

    return {
        "answer": response.text or "Aucune réponse générée.",
        "tool_name": "",
        "tool_args": {},
    }



# 6. Exécution de l'outil choisi
def execute_tool(state: AgentState) -> dict:
    name = state["tool_name"]
    history = list(state["tool_history"])
    args = state["tool_args"]

    if not name or name not in TOOLS:
        return {
            "answer": "Erreur : outil inconnu.",
            "tool_result": "",
            "last_tool_name": "",
            "tool_history": history,
            "steps": state["steps"],
            "tool_name": "",
            "tool_args": {},
        }

    try:
        result = TOOLS[name](**args)
        result_json = json.dumps(result, ensure_ascii=False)

        history.append(
            f"Outil : {name} | "
            f"Arguments : {json.dumps(args, ensure_ascii=False)} "
            f"| Résultat : {result_json}"
        )

        return {
            "tool_result": result_json,
            "tool_history": history,
            "last_tool_name": name,
            "steps": state["steps"] + 1,
            "tool_name": "",
            "tool_args": {},
        }

    except Exception as error:
        error_message = (
            f"Erreur lors de l'exécution de {name} : "
            f"{type(error).__name__}: {error}"
        )
        history.append(error_message)

        return {
            "tool_result": error_message,
            "tool_history": history,
            "last_tool_name": name,
            "steps": state["steps"] + 1,
            "tool_name": "",
            "tool_args": {},
        }


# 7. Décider si un autre outil est nécessaire
def route_after_tool(
    state: AgentState,
) -> Literal["ask_gemini", "summarize_result"]:
    if state["steps"] < MAX_TOOL_CALLS:
        return "ask_gemini"

    return "summarize_result"


# 8. Résumé final du résultat
def summarize_result(state: AgentState) -> dict:
    if state["answer"]:
        return {"answer": state["answer"]}

    if state["tool_history"]:
        return {
            "answer": (
                "Les appels d'outils sont terminés. "
                "Voici les résultats obtenus :\n"
                + "\n".join(state["tool_history"])
            )
        }

    return {
        "answer": "Je n'ai pas pu produire de résultat."
    }


# 9. Construction du workflow LangGraph
graph_builder = StateGraph(AgentState)

graph_builder.add_node("ask_gemini", ask_gemini)
graph_builder.add_node("execute_tool", execute_tool)
graph_builder.add_node("summarize_result", summarize_result)

graph_builder.add_edge(START, "ask_gemini")

graph_builder.add_conditional_edges(
    "ask_gemini",
    lambda state: (
        "execute_tool"
        if state["tool_name"] and state["steps"] < MAX_TOOL_CALLS
        else "summarize_result"
    ),
    {
        "execute_tool": "execute_tool",
        "summarize_result": "summarize_result",
    },
)

graph_builder.add_conditional_edges(
    "execute_tool",
    route_after_tool,
    {
        "ask_gemini": "ask_gemini",
        "summarize_result": "summarize_result",
    },
)

graph_builder.add_edge("summarize_result", END)

agent = graph_builder.compile()


# 10. Fonction publique utilisée par les tests et FastAPI
def run_agent(task: str) -> str:
    result = agent.invoke({
        "task": task,
        "answer": "",
        "tool_name": "",
        "tool_args": {},
        "tool_result": "",
        "tool_history": [],
        "last_tool_name": "",
        "steps": 0,
    })

    return result["answer"]



