from pathlib import Path
import json

from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR

DATASET_PATH = Path("dataset")
OUTPUT_PATH = Path("output")

OUTPUT_PATH.mkdir(exist_ok=True)

ocr = PaddleOCR(
lang="en",
device="gpu",
use_doc_orientation_classify=False,
use_doc_unwarping=False,
use_textline_orientation=False,
)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT = ImageFont.truetype(FONT_PATH, 18)

dataset = [
path
for path in DATASET_PATH.iterdir()
if path.is_file()
]

for image_path in dataset[:3]:

    print(f"Processing: {image_path.name}")

    results = ocr.predict(str(image_path))

    for result in results:

        data = result.json

        if isinstance(data, str):
            data = json.loads(data)

        data = data["res"]

        texts = data["rec_texts"]
        scores = data["rec_scores"]
        boxes = data["rec_polys"]

        # Keep only detections with confidence > 0.8
        detections = [
            (box, text, float(score))
            for box, text, score in zip(boxes, texts, scores)
            if float(score) > 0.8
        ]

        print(f"Detected {len(detections)} text regions")

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        for box, text, score in detections:

            print(f"  {score:.4f}  {text}")

            points = [
                (int(x), int(y))
                for x, y in box
            ]

            x = min(point[0] for point in points)
            y = min(point[1] for point in points)

            # Draw bounding box
            draw.line(
                points + [points[0]],
                fill="red",
                width=3,
            )

            # Draw text and confidence
            label = f"{text} ({score:.2f})"

            bbox = draw.textbbox(
                (x, y),
                label,
                font=FONT,
            )

            padding = 4

            draw.rectangle(
                (
                    bbox[0] - padding,
                    bbox[1] - padding,
                    bbox[2] + padding,
                    bbox[3] + padding,
                ),
                fill="red",
            )

            draw.text(
                (x, y),
                label,
                fill="white",
                font=FONT,
            )

        output_file = OUTPUT_PATH / image_path.name
        image.save(output_file)

