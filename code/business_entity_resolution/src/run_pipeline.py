#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from src.train import main as train_main
from src.infer import run_inference

def main():
    print("="*70)
    print("END-TO-END BUSINESS ENTITY RESOLUTION PIPELINE")
    print("="*70)
    
    model_path = os.path.join(os.path.dirname(__file__), "matching_model.joblib")
    if not os.path.isfile(model_path):
        print("\nStep 1: Training Entity Matching Model...")
        train_main()
    else:
        print(f"\nStep 1: Found existing model artifact at {model_path}.")
        
    print("\nStep 2: Running Inference on Test Dataset...")
    run_inference(test_dir="dataset/test", output_dir="output", model_path=model_path)
    
    print("\nPipeline execution complete!")

if __name__ == "__main__":
    main()
