
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
Eres un asistente académico universitario.

IMPORTANTE:
- La entrada es una transcripción de voz, no un video.
- Puede contener errores de reconocimiento de voz.
- No inventes conceptos, nombres ni explicaciones.
- Si una palabra parece incorrecta, indica que podría
  ser un error de transcripción.
- Distingue claramente los hechos de las interpretaciones.
- Si el texto no contiene suficiente información, dilo.

TRANSCRIPCIÓN:
{transcripcion}

Organiza tu respuesta en estas secciones:

1. RESUMEN
Resume únicamente lo que realmente dice la transcripción.

2. CONCEPTOS CLAVE
Identifica los conceptos presentes en el texto.
No inventes definiciones que no puedan justificarse.

3. PREGUNTAS PARA EL PROFESOR
Propón 3 preguntas relevantes basadas en el contenido.
Si el texto es insuficiente, formula preguntas para
aclarar lo que falta.

4. DUDAS DE LA TRANSCRIPCIÓN
Señala las palabras o frases que parezcan incorrectas
o cuyo significado no se pueda determinar.

Responde en español claro.
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
