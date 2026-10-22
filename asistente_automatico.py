
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import requests
import numpy as np
import queue
import threading
from datetime import datetime

FRECUENCIA = 48000
MICROFONO = 15
DURACION = 8
FRAGMENTOS_POR_ANALISIS = 4

cola_audio = queue.Queue()
cola_texto = queue.Queue()

transcripcion_completa = []
fragmentos_nuevos = 0
numero = 1
detener = threading.Event()
bloqueo_transcripcion = threading.Lock()


print("Cargando modelo de reconocimiento de voz...")

modelo = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

print("Modelo de voz listo.")
print("El asistente academico esta preparado.\n")


def recibir_audio(indata, frames, tiempo, estado):
    if estado:
        print("Aviso de audio:", estado)

    cola_audio.put(indata.copy())


def guardar_transcripcion():
    with bloqueo_transcripcion:
        texto = " ".join(transcripcion_completa)

    with open("transcripcion.txt", "w", encoding="utf-8") as archivo:
        archivo.write(texto)



def trabajador_audio():
    global numero

    muestras_necesarias = FRECUENCIA * DURACION
    buffer_audio = np.empty((0, 1), dtype=np.float32)

    while not detener.is_set() or not cola_audio.empty():
        try:
            audio = cola_audio.get(timeout=0.5)
        except queue.Empty:
            continue

        buffer_audio = np.concatenate(
            (buffer_audio, audio),
            axis=0
        )

        while len(buffer_audio) >= muestras_necesarias:
            audio_completo = buffer_audio[:muestras_necesarias]
            buffer_audio = buffer_audio[muestras_necesarias:]

            nombre_archivo = f"fragmento_{numero}.wav"
            sf.write(nombre_archivo, audio_completo, FRECUENCIA)

            print(f"\nProcesando fragmento {numero}...")

            try:
                import time
                inicio = time.time()
                segmentos, informacion = modelo.transcribe(
                nombre_archivo,
                language="es",
                vad_filter=True,
                initial_prompt="Estamos desarrollando un asistente académico con Python. Los términos técnicos son Whisper, Gemma 3, Ollama, transcripción de audio y asistente académico."
            )

                texto = " ".join(
                    segmento.text.strip()
                    for segmento in segmentos
                ).strip()
                print(f"Tiempo de transcripcion: {time.time() - inicio:.2f} segundos")

                if texto:
                    print(f"\nPROFESOR: {texto}\n")

                    with bloqueo_transcripcion:
                        transcripcion_completa.append(texto)

                    guardar_transcripcion()
                    cola_texto.put(texto)
                else:
                    print("No se detecto voz en este fragmento.")

            except Exception as error:
                print("Error al transcribir:", error)

            finally:
                import os

                if os.path.exists(nombre_archivo):
                    os.remove(nombre_archivo)

            numero += 1

    if len(buffer_audio) > 0:
        nombre_archivo = f"fragmento_{numero}.wav"
        sf.write(nombre_archivo, buffer_audio, FRECUENCIA)

        try:
            segmentos, informacion = modelo.transcribe(
            nombre_archivo,
            language="es",
            vad_filter=True,
            initial_prompt="Estamos desarrollando un asistente académico con Python. Los términos técnicos son Whisper, Gemma 3, Ollama, transcripción de audio y asistente académico."
        )

            texto = " ".join(
                segmento.text.strip()
                for segmento in segmentos
            ).strip()

            if texto:
                print(f"\nPROFESOR: {texto}\n")

                with bloqueo_transcripcion:
                    transcripcion_completa.append(texto)

                guardar_transcripcion()
                cola_texto.put(texto)

        except Exception as error:
            print("Error al transcribir el ultimo fragmento:", error)

        finally:
            import os

            if os.path.exists(nombre_archivo):
                os.remove(nombre_archivo)



def analizar_clase(textos):
    texto_reciente = " ".join(textos)

    prompt = f"""
Eres un asistente academico universitario.
Analiza solamente la transcripcion proporcionada.

La transcripcion puede contener palabras omitidas,
errores de reconocimiento y frases incompletas.

Reglas:
- No inventes explicaciones atribuidas al profesor.
- Distingue lo que aparece en el texto de tus inferencias.
- Si falta contexto, dilo claramente.
- Si el fragmento es insuficiente, no inventes contenido.
- No afirmes que la transcripcion es perfecta si hay
  frases incompletas o ideas que quedan sin terminar.

TRANSCRIPCION:
\"\"\"
{texto_reciente}
\"\"\"

Responde en español usando estas secciones:

1. RESUMEN BREVE
2. CONCEPTOS MENCIONADOS
3. PREGUNTAS UTILES PARA EL PROFESOR
4. POSIBLES ERRORES O FRASES INCOMPLETAS

En la ultima seccion, cita las frases dudosas del texto.
Si no encuentras ninguna, indica "Ninguna evidente".
"""

    print("\nAnalizando la clase en segundo plano...")

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

        fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(
            "sugerencias_clase.txt",
            "a",
            encoding="utf-8"
        ) as archivo:
            archivo.write("\n\n" + "=" * 50 + "\n")
            archivo.write("ANALISIS: " + fecha + "\n\n")
            archivo.write(resultado)

        print("\nSugerencias guardadas.")

    except requests.exceptions.RequestException as error:
        print("\nError al consultar Ollama:", error)


def trabajador_analisis():
    textos_pendientes = []

    while not detener.is_set() or not cola_texto.empty():
        try:
            texto = cola_texto.get(timeout=0.5)
        except queue.Empty:
            continue

        textos_pendientes.append(texto)

        if len(textos_pendientes) >= FRAGMENTOS_POR_ANALISIS:
            analizar_clase(textos_pendientes)
            textos_pendientes = []

    if textos_pendientes:
        analizar_clase(textos_pendientes)


hilo_audio = threading.Thread(target=trabajador_audio)
hilo_analisis = threading.Thread(target=trabajador_analisis)

try:
    print("Escuchando la clase.")
    print("Presiona Ctrl + C para detener.\n")

    with sd.InputStream(
        samplerate=FRECUENCIA,
        channels=1,
        dtype="float32",
        device=MICROFONO,
        callback=recibir_audio
    ):
        hilo_audio.start()
        hilo_analisis.start()

        while True:
            sd.sleep(200)

except KeyboardInterrupt:
    print("\nDeteniendo el asistente...")

except Exception as error:
    print("\nError durante la grabacion:", error)

finally:
    detener.set()

    if hilo_audio.is_alive():
        hilo_audio.join()

    if hilo_analisis.is_alive():
        hilo_analisis.join()

    guardar_transcripcion()

    print("\nTranscripcion guardada en transcripcion.txt")
    print("Sugerencias guardadas en sugerencias_clase.txt")
    print("Asistente detenido.")
