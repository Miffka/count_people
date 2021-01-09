import argparse

import cv2
import numpy as np
import pafy
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch

from config import model_config


def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in plt.rcParams["axes.prop_cycle"].by_key()["color"]]


def display(preds, conf_threshold=0.0):
    colors = color_list()
    counts = {"left": 0, "right": 0, "total": 0}
    im_h, im_w = preds.imgs[0].shape[:2]
    for img, pred in zip(preds.imgs, preds.pred):
        if pred is not None:
            img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
            for *box, conf, class_id in pred:  # xyxy, confidence, class
                if class_id == 0 and conf > conf_threshold:
                    ImageDraw.Draw(img).rectangle(box, width=4, outline=colors[int(class_id) % 10])  # plot
                    ImageDraw.Draw(img).text((box[0], box[1] - 10), f"{conf:.2f}", colors[int(class_id) % 10])

                    counts["total"] += 1
                    if box[0] < im_w // 2:
                        counts["left"] += 1
                    else:
                        counts["right"] += 1
    ImageDraw.Draw(img).text((10, 20), f"Left: {counts['left']}", (0, 255, 0))
    ImageDraw.Draw(img).text((im_w // 2 + 10, 20), f"Right: {counts['right']}", (0, 255, 0))
    return np.asarray(img), counts


def get_draw_count(source_name):
    source_url = VIDEO_ULRLS[source_name]["url"]
    if VIDEO_ULRLS[source_name]["use_pafy"]:
        vPafy = pafy.new(source_url)
        play = vPafy.getbest(preftype="mp4")
        url = play.url
    else:
        url = source_url

    cap = cv2.VideoCapture(url)

    ret, frame = cap.read()
    img = Image.fromarray(frame)
    with torch.no_grad():
        predicts = model([img])
    frame, counts = display(predicts)

    cap.release()
    return frame, counts


VIDEO_ULRLS = {
    "gostiny": {"url": "https://youtu.be/wCcMcaiRbhM", "use_pafy": True},
    "anichkov": {"url": "https://youtu.be/jbqT0fTj088", "use_pafy": True},
    "slavy": {"url": "http://93.100.5.209:8093/mjpg/video.mjpg", "use_pafy": False},
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visulaize real-time people count")
    parser.add_argument("--source_name", default="slavy", help="String with name of source")
    args = parser.parse_args()

    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model.to(model_config.device)
    model.eval()

    while True:
        frame, counts = get_draw_count(args.source_name)

        cv2.imshow("frame", frame)

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
