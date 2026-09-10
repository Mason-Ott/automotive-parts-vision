from pathlib import Path

import cv2
from paddleocr import PaddleOCR


ocr = PaddleOCR(
    lang="en",
    device="cpu",
    enable_mkldnn=False,
)

dataset_path = Path("dataset")

dataset = [
    path for path in dataset_path.iterdir()
    if path.is_file()
]

for diagram_path in dataset:
    print(f"\nProcessing: {diagram_path.name}")

    diagram = cv2.imread(str(diagram_path))

    result = ocr.predict(diagram)

    for res in result:
        print(res)