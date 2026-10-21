
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import os

FRECUENCIA = 48000
MICROFONO = 15
DURACION = 8

print("Cargando modelo de inteligencia artificial...")

modelo = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

print("Modelo listo.")
print("El asistente está escuchando.")
print("Presiona Ctrl + C para detenerlo.\n")

numero = 1
transcripcion_completa = []

try:
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

        archivo = f"fragmento_{numero}.wav"
        sf.write(archivo, audio, FRECUENCIA)

        segmentos, informacion = modelo.transcribe(
            archivo,
            language="es",
            vad_filter=True
        )

        texto = " ".join(
            segmento.text.strip()
            for segmento in segmentos
        ).strip()

        if texto:
            print(f"\nTÚ: {texto}\n")
            transcripcion_completa.append(texto)
        else:
            print("No se detectó voz.\n")

        os.remove(archivo)
        numero += 1

except KeyboardInterrupt:
    print("\nAsistente detenido.")

    texto_final = " ".join(transcripcion_completa)

    print("\n===== TRANSCRIPCIÓN COMPLETA =====\n")
    print(texto_final)

    with open("transcripcion.txt", "w", encoding="utf-8") as archivo:
        archivo.write(texto_final)

    print("\nLa transcripción se guardó en transcripcion.txt")
