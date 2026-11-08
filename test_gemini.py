
import os

from dotenv import load_dotenv
from google import genai


def test_gemini_connection():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY est absente du fichier .env"
        )

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="Explique en une phrase ce qu'est un agent IA."
    )

    assert response.text is not None
    assert len(response.text.strip()) > 0

    print(response.text)


if __name__ == "__main__":
    test_gemini_connection()
