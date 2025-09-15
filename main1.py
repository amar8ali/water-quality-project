# Entry point for Water Quality AI Project

from src.data.preprocessing import preprocess_data
from src.models.train_models import train_and_evaluate

def main():
    print("Welcome to the Water Quality AI Project!")
    # Preprocess data
    processed = preprocess_data('data/water_potability.csv', 'data/water_potability_processed.csv')
    # Train and evaluate models
    results = train_and_evaluate(processed)
    for model, metrics in results.items():
        print(f"\nModel: {model}")
        print(f"Accuracy: {metrics['accuracy']:.2f}")
        print(metrics['report'])

if __name__ == "__main__":
    main()
