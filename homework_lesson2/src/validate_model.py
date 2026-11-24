import sys

from evaluate import get_metrics, load_params


def validate_model():
    accuracy = get_metrics()["accuracy"]
    accuracy_min = load_params()["accuracy_min"]

    print(f"Accuracy: {accuracy:.4f}, minimum: {accuracy_min:.4f}")
    if accuracy < accuracy_min:
        print("Model validation failed!")
        sys.exit(1)

    print("Model validation passed!")


if __name__ == "__main__":
    validate_model()
