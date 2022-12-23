import cv2
import argparse
from pathlib import Path


def main(args):
    working_dir = Path(args.working_dir)
    for image_path in working_dir.rglob("*.png"):
        
        if "female" in image_path.stem or "male" in image_path.stem:
            continue
        
        image = cv2.imread(str(image_path))
        image = cv2.resize(image, (512, 512))
        cv2.imshow("image", image)
        key = cv2.waitKey(0)

        if key == ord("q"):
            break

        if key == ord("m"):
            image_path.rename(image_path.as_posix().replace(".png", "_male.png"))

        if key == ord("f"):
            image_path.rename(image_path.as_posix().replace(".png", "_female.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("working_dir", type=str)
    args = parser.parse_args()
    main(args)
