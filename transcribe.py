# transcribe.py
import os
import torch
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel
import json
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verificar_cuda():
    if torch.cuda.is_available():
        logging.info(f"✅ CUDA detectado. Usando GPU: {torch.cuda.get_device_name(0)}")
        return "cuda", "float16"
    else:
        logging.warning("⚠️ CUDA no está disponible. El proceso se ejecutará en la CPU.")
        return "cpu", "int8"

def extraer_audio(video_path: Path, audio_output_path: Path):
    if audio_output_path.exists():
        logging.info(f"El archivo de audio '{audio_output_path}' ya existe. Saltando extracción.")
        return
    try:
        logging.info(f"Extrayendo audio de '{video_path}'...")
        with VideoFileClip(str(video_path)) as video:
            video.audio.write_audiofile(str(audio_output_path))
        logging.info(f"Audio extraído y guardado en '{audio_output_path}'")
    except Exception as e:
        logging.error(f"Error al extraer el audio: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Paso 1: Transcribe el audio de un video.")
    parser.add_argument("video_path", type=str, help="Ruta al archivo de video .mp4 de entrada.")
    parser.add_argument("--modelo_whisper", type=str, default="medium", help="Modelo de Whisper a usar.")
    args = parser.parse_args()

    start_time = time.time()
    device, compute_type = verificar_cuda()

    video_input_path = Path(args.video_path)
    if not video_input_path.is_file():
        logging.error(f"El archivo de video '{video_input_path}' no existe.")
        return

    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    base_name = video_input_path.stem
    temp_audio_path = output_dir / f"{base_name}_audio.wav"
    json_output_path = output_dir / f"{base_name}_transcription.json" # <-- Salida de este script

    try:
        extraer_audio(video_input_path, temp_audio_path)
        
        logging.info(f"Cargando modelo de transcripción Whisper '{args.modelo_whisper}' en {device}...")
        model = WhisperModel(args.modelo_whisper, device=device, compute_type=compute_type)
        
        logging.info("Iniciando transcripción de audio...")
        segments, _ = model.transcribe(str(temp_audio_path), beam_size=5, language="en")
        
        transcripcion_completa = []
        for segment in tqdm(segments, desc="Transcripción"):
            transcripcion_completa.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text
            })
        
        if not transcripcion_completa:
            logging.error("La transcripción no produjo ningún segmento. Abortando.")
            return

        logging.info(f"Guardando transcripción en '{json_output_path}'...")
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(transcripcion_completa, f, indent=4, ensure_ascii=False)

    except Exception as e:
        logging.error(f"Ocurrió un error en el script de transcripción: {e}")
        return

    end_time = time.time()
    logging.info(f"🚀 Paso 1 (Transcripción) completado en {end_time - start_time:.2f} segundos.")
    logging.info(f"Los datos de la transcripción están en: '{json_output_path}'")

if __name__ == "__main__":
    main()