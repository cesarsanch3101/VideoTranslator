# translate.py
import torch
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
import json
import time
import ffmpeg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verificar_cuda():
    if torch.cuda.is_available():
        logging.info(f"✅ CUDA detectado. Usando GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        logging.warning("⚠️ CUDA no está disponible. El proceso se ejecutará en la CPU.")
        return "cpu"

def format_time(seconds):
    millis = int((seconds - int(seconds)) * 1000)
    return time.strftime('%H:%M:%S', time.gmtime(seconds)) + f',{millis:03d}'

def generar_srt(segmentos_traducidos: list, srt_path: Path):
    logging.info(f"Generando archivo de subtítulos .srt en '{srt_path}'...")
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segmentos_traducidos):
            f.write(f"{i + 1}\n")
            f.write(f"{format_time(segment['start'])} --> {format_time(segment['end'])}\n")
            f.write(f"{segment['translated_text']}\n\n")

def incrustar_subtitulos(video_original: Path, srt_path: Path, output_path: Path):
    logging.info(f"Incrustando subtítulos en el video final: '{output_path}'...")
    srt_escaped_path = str(srt_path).replace('\\', '/').replace(':', '\\:')
    
    input_video = ffmpeg.input(str(video_original))
    output_video = ffmpeg.output(
        input_video.video,
        input_video.audio,
        str(output_path),
        vf=f"subtitles='{srt_escaped_path}'",
        acodec='copy', vcodec='libx264', crf=23, preset='medium'
    )
    
    try:
        output_video.run(overwrite_output=True, quiet=True)
        logging.info("Video con subtítulos incrustados guardado exitosamente.")
    except ffmpeg.Error as e:
        logging.error("Error de ffmpeg al incrustar subtítulos:")
        logging.error(e.stderr.decode())
        raise

def main():
    parser = argparse.ArgumentParser(description="Paso 2: Traduce segmentos y genera el video final.")
    parser.add_argument("json_path", type=str, help="Ruta al archivo .json con la transcripción.")
    parser.add_argument("original_video_path", type=str, help="Ruta al video .mp4 original.")
    parser.add_argument("--modelo_traduccion", type=str, default="Helsinki-NLP/opus-mt-en-es", help="Modelo de traducción de Hugging Face.")
    args = parser.parse_args()

    start_time = time.time()
    device = verificar_cuda()

    json_input_path = Path(args.json_path)
    video_input_path = Path(args.original_video_path)
    if not json_input_path.is_file() or not video_input_path.is_file():
        logging.error(f"No se encuentra el archivo JSON '{json_input_path}' o el video '{video_input_path}'.")
        return

    output_dir = Path("./output")
    base_name = video_input_path.stem
    srt_output_path = output_dir / f"{base_name}_subtitulos.srt"
    video_output_path = output_dir / f"{base_name}_subtitulado.mp4"

    try:
        with open(json_input_path, 'r', encoding='utf-8') as f:
            segmentos_transcritos = json.load(f)

        logging.info(f"Cargando modelo de traducción '{args.modelo_traduccion}'...")
        translator = pipeline("translation", model=args.modelo_traduccion, device=0 if device == 'cuda' else -1)
        
        logging.info("Iniciando traducción de segmentos...")
        for segment in tqdm(segmentos_transcritos, desc="Traducción"):
            try:
                translation = translator(segment.get('text'))
                segment['translated_text'] = translation[0]['translation_text']
            except Exception as e:
                segment['translated_text'] = "[Error en traducción]"

        generar_srt(segmentos_transcritos, srt_output_path)
        incrustar_subtitulos(video_input_path, srt_output_path, video_output_path)

    except Exception as e:
        logging.error(f"Ocurrió un error en el script de traducción: {e}")
        return
        
    end_time = time.time()
    logging.info(f"🚀 Paso 2 (Traducción y Ensamblaje) completado en {end_time - start_time:.2f} segundos.")
    logging.info(f"🎉 ¡Video final listo en: '{video_output_path}'!")

if __name__ == "__main__":
    main()