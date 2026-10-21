
import requests

print("Leyendo la transcripción de la clase...")

try:
    with open("transcripcion.txt", "r", encoding="utf-8") as archivo:
        transcripcion = archivo.read().strip()

    if not transcripcion:
        print("La transcripción está vacía.")
        raise SystemExit

except FileNotFoundError:
    print("No se encontró transcripcion.txt.")
    print("Primero debes ejecutar asistente_voz.py y grabar una clase.")
    raise SystemExit

print("Enviando la transcripción a la inteligencia artificial...")

prompt = f"""
Actúa como un asistente académico universitario.

Analiza la siguiente transcripción de una clase.

TRANSCRIPCIÓN:
{transcripcion}

Realiza estas tareas:

1. Resume las ideas principales.
2. Identifica los conceptos importantes.
3. Formula 3 preguntas inteligentes que el estudiante
   podría hacer al profesor para participar en clase.
4. Explica cualquier concepto difícil con palabras sencillas.

Usa únicamente la información disponible en la transcripción.
Si falta información, indícalo en lugar de inventarla.
Responde en español y organiza la respuesta por secciones.
"""

try:
    respuesta = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma3:1b",
            "prompt": prompt,
            "stream": False
        },
        timeout=300
    )

    respuesta.raise_for_status()

    resultado = respuesta.json()["response"]

    print("\n===== ANÁLISIS DE LA CLASE =====\n")
    print(resultado)

    with open("analisis_clase.txt", "w", encoding="utf-8") as archivo:
        archivo.write(resultado)

    print("\nEl análisis se guardó en analisis_clase.txt")

except requests.exceptions.ConnectionError:
    print("No se pudo conectar con Ollama.")
    print("Comprueba que Ollama esté ejecutándose.")

except requests.exceptions.Timeout:
    print("La IA tardó demasiado en responder.")

except requests.exceptions.RequestException as error:
    print("Ocurrió un error al consultar la IA:", error)
