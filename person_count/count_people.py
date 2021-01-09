import argparse
import os
import os.path as osp
import time

import cv2
import pafy
import pandas as pd
from PIL import Image
import torch

from config import model_config
from draw_people import VIDEO_ULRLS


def count_people(preds, conf_threshold=0.0):
    counts = {"left": [0], "right": [0], "total": [0]}
    im_h, im_w = preds.imgs[0].shape[:2]
    for pred in preds.pred:
        if pred is not None:
            for *box, conf, class_id in pred:  # xyxy, confidence, class
                if class_id == 0 and conf > conf_threshold:
                    counts["total"][0] += 1
                    if box[0] < im_w // 2:
                        counts["left"][0] += 1
                    else:
                        counts["right"][0] += 1
    return counts


def get_count(source_name):
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
    counts = count_people(predicts)
    counts["time"] = [time.strftime("%Y-%m-%d %H:%M:%S")]
    counts["source_name"] = [source_name]

    cap.release()
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Count people and save info to file")
    parser.add_argument("--task_name", default="all_logs", help="Name of the task and output file")
    parser.add_argument(
        "--output_dir", default=osp.join(model_config.data_dir, "processed"), help="Path to directory to log everything"
    )
    args = parser.parse_args()

    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model.to(model_config.device)
    model.eval()

    out_file = osp.join(args.output_dir, f"{args.task_name}.csv")

    if osp.exists(out_file):
        df = pd.read_csv(out_file)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        df = pd.DataFrame(columns=["time", "source_name", "total", "left", "right"])

    while True:
        for source_name in VIDEO_ULRLS.keys():
            counts = get_count(source_name)
            df_i = pd.DataFrame(counts)
            df = pd.concat([df, df_i])
            time.sleep(1)
            df.to_csv(out_file, index=False)
