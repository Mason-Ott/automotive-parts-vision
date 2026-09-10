from pathlib import Path
import cv2
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import matplotlib.pyplot as plt

# Initialize PaddleOCR
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    device="cpu",
    enable_mkldnn=False,
)

dataset_path = Path("dataset")
dataset = [path for path in dataset_path.iterdir() if path.is_file()]

for diagram_path in dataset[:3]:
    print(f"\nProcessing: {diagram_path.name}")

    # Run OCR using ocr.ocr() method
    result = ocr.ocr(str(diagram_path), cls=True)
    
    # Extract the first page result
    result = result[0]
    
    # Extract boxes, texts, and scores
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    
    print(f"Detected {len(boxes)} text regions")
    
    # Load original image
    image = Image.open(diagram_path).convert('RGB')
    
    # Draw OCR results
    im_show = draw_ocr(
        image, 
        boxes, 
        txts, 
        scores,
        font_path='C:\\Windows\\Fonts\\arial.ttf'
    )
    
    # Display
    plt.figure(figsize=(16, 10))
    plt.imshow(im_show)
    plt.title(diagram_path.name)
    plt.axis("off")
    plt.tight_layout()
    plt.show()