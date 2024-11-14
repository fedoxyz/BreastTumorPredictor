import argparse
from src.train import train_model
from src.inference import predict
import yaml

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "inference"], required=True)
    parser.add_argument("--image_path", type=str, help="Path to the image for inference")
    parser.add_argument("--model_path", type=str, help="Path to the saved model")

    args = parser.parse_args()

    if args.mode == "train":
        train_model(config)
    elif args.mode == "inference" and args.image_path and args.model_path:
        mask = predict(args.image_path, args.model_path, img_size=[256, 256])
        print("Inference done!")
