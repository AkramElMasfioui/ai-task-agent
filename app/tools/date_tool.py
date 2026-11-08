
from datetime import datetime


def get_current_date() -> str:
    """Retourne la date et l'heure actuelles."""
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")
