from app.agent import execute_tool, summarize_result, route_after_tool
from app.tools.calculator import add, multiply
from app.tools.date_tool import get_current_date



def test_addition():
    assert add(120, 250) == 370


def test_multiplication():
    assert multiply(12, 5) == 60


def test_multiplication_resultat_addition():
    resultat = add(120, 250)
    assert multiply(resultat, 2) == 740


def test_execution_outil_add():
    state = {
        "task": "Calcule 120 + 250",
        "answer": "",
        "tool_name": "add",
        "tool_args": {"a": 120, "b": 250},
        "tool_result": "",
        "tool_history": [],
        "last_tool_name": "",
        "steps": 0,
    }

    result = execute_tool(state)

    assert result["tool_result"] == '{"result": 370}'
    assert result["steps"] == 1
    assert result["last_tool_name"] == "add"
    assert len(result["tool_history"]) == 1


def test_resume_resultats():
    state = {
        "task": "Calcul",
        "answer": "",
        "tool_name": "",
        "tool_args": {},
        "tool_result": '{"result": 370}',
        "tool_history": [
            'Outil : add | Arguments : {"a": 120, "b": 250} | Résultat : {"result": {"result": 370}}'
        ],
        "last_tool_name": "add",
        "steps": 1,
    }

    result = summarize_result(state)

    assert "370" in result["answer"]

def test_execution_outil_inconnu():
    state = {
        "task": "Test",
        "answer": "",
        "tool_name": "outil_inexistant",
        "tool_args": {},
        "tool_result": "",
        "tool_history": [],
        "last_tool_name": "",
        "steps": 0,
    }

    result = execute_tool(state)

    assert "inconnu" in result["answer"].lower()
    assert result["steps"] == 0


def test_limite_appels_outils():
    state = {
        "task": "Test",
        "answer": "",
        "tool_name": "add",
        "tool_args": {"a": 10, "b": 5},
        "tool_result": "",
        "tool_history": [],
        "last_tool_name": "",
        "steps": 3,
    }

    result = route_after_tool(state)

    assert result == "summarize_result"

def test_get_current_date():
    result = get_current_date()

    assert isinstance(result, str)
    assert len(result) > 0
