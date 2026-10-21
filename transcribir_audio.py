
from faster_whisper import WhisperModel

modelo = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("Cargando el audio...")

segmentos, informacion = modelo.transcribe(
    "grabacion.wav",
    language="es"
)

print("\nTranscripción:\n")

for segmento in segmentos:
    print(segmento.text)

print("\nTranscripción terminada.")
