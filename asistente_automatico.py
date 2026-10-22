
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import requests
import os
from datetime import datetime

FRECUENCIA = 48000
MICROFONO = 15
DURACION = 15
FRAGMENTOS_POR_ANALISIS = 4

print("Cargando modelo de reconocimiento de voz...")

modelo = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

print("Modelo de voz listo.")
print("El asistente academico esta preparado.\n")

transcripcion_completa = []
fragmentos_nuevos = 0
numero = 1


def guardar_transcripcion():
    texto = " ".join(transcripcion_completa)

    with open("transcripcion.txt", "w", encoding="utf-8") as archivo:
        archivo.write(texto)


def analizar_clase():
    if not transcripcion_completa:
        return

    texto_reciente = " ".join(transcripcion_completa[-8:])

    prompt = f"""
Eres un asistente academico universitario.

Analiza exclusivamente la transcripcion proporcionada.
Puede contener errores del reconocimiento de voz.

REGLAS:
- No inventes datos ni afirmes que el profesor dijo algo
  que no aparece en la transcripcion.
- Si falta informacion, dilo claramente.
- Si una frase parece incorrecta, advierte que podria
  ser un error de reconocimiento de voz.
- No repitas preguntas que no sean necesarias.
- Diferencia la informacion explicita de tus inferencias.

TRANSCRIPCION RECIENTE:
\"\"\"
{texto_reciente}
\"\"\"

Responde en español con estas secciones:

1. RESUMEN BREVE
Resume las ideas principales de este fragmento.

2. CONCEPTOS CLAVE
Enumera los conceptos mencionados y explica solo lo
que pueda justificarse con el texto.

3. PREGUNTAS PARA EL PROFESOR
Propón hasta 3 preguntas útiles para aclarar el tema.
No inventes preguntas sobre contenidos no mencionados.

4. POSIBLES ERRORES DE TRANSCRIPCION
Indica las frases confusas o escribe "Ninguno evidente"
si no encuentras problemas claros.

Si el fragmento contiene muy poca informacion,
explica esa limitacion en lugar de inventar contenido.
"""

    print("\nAnalizando lo explicado recientemente...")

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

        print("\n===== SUGERENCIAS ACADEMICAS =====\n")
        print(resultado)

        with open(
            "sugerencias_clase.txt",
            "a",
            encoding="utf-8"
        ) as archivo:
            fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            archivo.write("\n\n" + "=" * 50 + "\n")
            archivo.write("ANALISIS: " + fecha + "\n\n")
            archivo.write(resultado)

        print("\nSugerencias guardadas en sugerencias_clase.txt")

    except requests.exceptions.ConnectionError:
        print("\nNo se pudo conectar con Ollama.")
        print("Comprueba que Ollama este ejecutandose.")

    except requests.exceptions.Timeout:
        print("\nLa IA tardo demasiado en responder.")

    except requests.exceptions.RequestException as error:
        print("\nError al consultar Ollama:", error)


try:
    print("Escuchando la clase. Presiona Ctrl + C para detener.\n")

    while True:
        print(f"Grabando fragmento {numero}...")

        audio = sd.rec(
            int(DURACION * FRECUENCIA),
            samplerate=FRECUENCIA,
            channels=1,
            dtype="float32",
            device=MICROFONO
        )

        sd.wait()

        nombre_archivo = f"fragmento_{numero}.wav"
        sf.write(nombre_archivo, audio, FRECUENCIA)

        segmentos, informacion = modelo.transcribe(
            nombre_archivo,
            language="es",
            vad_filter=True
        )

        texto = " ".join(
            segmento.text.strip()
            for segmento in segmentos
        ).strip()

        os.remove(nombre_archivo)

        if texto:
            print(f"\nPROFESOR: {texto}\n")
            transcripcion_completa.append(texto)
            fragmentos_nuevos += 1
            guardar_transcripcion()
        else:
            print("No se detecto voz.\n")

        numero += 1

        if fragmentos_nuevos >= FRAGMENTOS_POR_ANALISIS:
            analizar_clase()
            fragmentos_nuevos = 0

except KeyboardInterrupt:
    print("\n\nDeteniendo el asistente...")

finally:
    guardar_transcripcion()
    print("Transcripcion guardada en transcripcion.txt")
    print("Asistente detenido.")
