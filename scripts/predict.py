import argparse
import pandas as pd
from src.predict import load_model, predict

def main(args):
    # Load the data
    data = pd.read_csv(args.input_data)
    
    # Load the model
    model = load_model(args.model_path)
    
    # Make predictions
    predictions, probabilities = predict(model, data)
    
    # Save predictions to output file
    results = pd.DataFrame({
        'predictions': predictions,
        'probabilities': probabilities
    })
    results.to_csv(args.output_file, index=False)
    print(f"Predictions saved to {args.output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make predictions using a trained model.')
    parser.add_argument('--input-data', type=str, required=True, help='Path to the input data CSV file.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--output-file', type=str, required=True, help='Path to save the predictions CSV file.')

    args = parser.parse_args()
    main(args)
