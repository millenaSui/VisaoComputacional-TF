"""
Módulo principal para processar imagens em um diretório específico. Ele verifica
a existência de imagens com extensões válidas, carrega o modelo YOLOE-26,
processa cada imagem encontrada e salva os resultados em um diretório de saída.
"""

import os
from ultralytics import YOLOE

from src.images import process_images

INPUT_DIR = './dataset'
OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

HUMAN_CLASSES = ["person", "woman", "man", "child", "baby", "human"]
DISTRACTOR_CLASSES = ["painting", "artwork", "picture frame", "landscape", "nature"]


def main():
    """
    Função principal que verifica a existência de imagens no diretório de entrada,
    carrega o modelo YOLOE-26, processa cada imagem encontrada e salva os resultados.
    """
    ext = ('.jpg', '.jpeg', '.png', '.bmp')
    imagens = []

    if os.path.exists(INPUT_DIR):
        for arquivo in os.listdir(INPUT_DIR):
            if arquivo.lower().endswith(ext):
                imagens.append(os.path.join(INPUT_DIR, arquivo))

    if not imagens:
        print(f"[ERRO] Nenhuma imagem encontrada na pasta {INPUT_DIR}.")
        return

    print("[INFO] Carregando modelo YOLOE-26...")
    model = YOLOE("yoloe-26l-seg.pt")
    model.set_classes(HUMAN_CLASSES + DISTRACTOR_CLASSES)

    print(f"[INFO] Encontradas {len(imagens)} imagens.")
    for img_path in sorted(imagens):
        print(f"[INFO] Processando {os.path.basename(img_path)}...")
        process_images(
            img_path,
            OUTPUT_DIR,
            model=model,
            num_human_classes=len(HUMAN_CLASSES),
        )


if __name__ == "__main__":
    main()