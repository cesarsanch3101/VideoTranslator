# download_model.py
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

MODEL_NAME = "Helsinki-NLP/opus-mt-en-es"

def download():
    """Downloads and caches the specified Hugging Face model."""
    try:
        logging.info(f"Iniciando la descarga del modelo: {MODEL_NAME}")
        logging.info("Esto puede tardar varios minutos la primera vez. Por favor, espera...")
        # This line triggers the download and caches the model
        pipeline("translation", model=MODEL_NAME)
        logging.info("✅ ¡Modelo descargado y guardado en caché exitosamente!")
    except Exception as e:
        logging.error(f"❌ Ocurrió un error durante la descarga: {e}")
        logging.error("Verifica tu conexión a internet y que no haya un firewall bloqueando la descarga.")

if __name__ == "__main__":
    download()