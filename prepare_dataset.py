import base64
from pathlib import Path

import cv2
import numpy as np
import orjson

COLORS = {"wall": (1, 1, 1), "door": (3, 3, 3), "window": (2, 2, 2)}
LABELS_PATH = Path("/Users/artem/datasets/cryme2022_dataset/train/object_detection")
OUTPUT_PATH = LABELS_PATH.parent / "full_dataset"
IMAGE_PATH = OUTPUT_PATH / "images"
MASK_PATH = OUTPUT_PATH / "annotations"
IMAGE_PATH.mkdir(exist_ok=True, parents=True)
MASK_PATH.mkdir(exist_ok=True, parents=True)


def draw_outer_rectangle(image, polygon):
    rect = cv2.minAreaRect(polygon)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, (0, 0, 255), -1)


def minimum_bounding_box(points):
    x_min = min(p[0] for p in points)
    y_min = min(p[1] for p in points)
    x_max = max(p[0] for p in points)
    y_max = max(p[1] for p in points)

    return x_min, y_min, x_max, y_max


def base64_to_npimage(image: str) -> np.ndarray:
    image = base64.b64decode(image)
    image = np.asarray(bytearray(image), dtype=np.uint8)
    image = cv2.imdecode(image, -1)
    return image


for json_file in LABELS_PATH.rglob("*.json"):
    # json_file = LABELS_PATH / "89df4ed1-3423-4a71-90c5-297588814877.json"
    with json_file.open(mode="r") as f:
        labeled_data = orjson.loads(f.read())

    base64_image = labeled_data["imageData"]
    shapes = labeled_data.get("shapes")
    image = base64_to_npimage(image=base64_image)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    mask = np.zeros(image.shape, dtype=np.uint8)
    if len(shapes) < 2:
        continue

    for shape in shapes:
        points = shape["points"]
        if len(points) < 2:
            continue
        label = shape["label"]
        shape_type = shape["shape_type"]
        if label in COLORS:

            color = COLORS.get(label)
            if shape_type == "polygon":
                coords = np.array([points]).astype(int)
                if label == "door":
                    x1, y1, x2, y2 = minimum_bounding_box(points)
                    cv2.rectangle(
                        mask, (int(x1), int(y1)), (int(x2), int(y2)), color, -1
                    )
                    # draw_outer_rectangle(mask, coords)
                else:
                    cv2.fillPoly(mask, [coords], color)
            elif shape_type == "rectangle":

                x1 = points[0][0]
                y1 = points[0][1]
                x2 = points[1][0]
                y2 = points[1][1]
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)

    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    image_save_path = IMAGE_PATH.joinpath(f"{json_file.stem}.jpg").__str__()
    mask_save_path = MASK_PATH.joinpath(f"{json_file.stem}.png").__str__()
    cv2.imwrite(image_save_path, image)
    cv2.imwrite(mask_save_path, mask)

    print(labeled_data["imagePath"])
