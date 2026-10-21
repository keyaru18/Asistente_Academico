
import sounddevice as sd
import soundfile as sf

frecuencia = 48000
duracion = 5
microfono = 15

print("Grabando... Habla ahora.")

audio = sd.rec(
    int(duracion * frecuencia),
    samplerate=frecuencia,
    channels=1,
    device=microfono
)

sd.wait()

sf.write("grabacion.wav", audio, frecuencia)

print("Grabación terminada.")
print("Archivo guardado como grabacion.wav")
