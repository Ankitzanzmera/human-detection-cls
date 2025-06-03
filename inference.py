import os
import torch
import argparse
import os.path as osp
import matplotlib.pyplot as plt

from human_detection_cls.prediction import PredictionPipeline

def pred_image(args):
    pred_pipeline = PredictionPipeline(args)
    output = pred_pipeline.pred()
    print(f"Model Predicted {output}...")

def main():
    parser = argparse.ArgumentParser(description = "Input for reference")
    parser.add_argument("--img_path", type=str, required=True, help="Enter path of Image")
    parser.add_argument("--device", action="store_true", help="To use GPU for inference")
    parser.add_argument("--save", action="store_true", help="To Save Output")
    
    args = parser.parse_args()

    if args.device:
        if torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    else:
        args.device = "cpu"

    if not osp.exists(args.img_path):
        raise ValueError("Given Image Does Not Exist...")
    
    if args.save:
        args.save = osp.join(os.getcwd(), "preds")
        os.makedirs(args.save, exist_ok=True)
        
        
    pred_image(args)

if __name__ == "__main__":
    main()