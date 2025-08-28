import argparse
from ultralytics import YOLO

def run_prediction(weights, source, save_dir):
    # Load the trained YOLO model
    model = YOLO(weights)

    # Run prediction on the input image
    results = model.predict(source=source, save=True, project=save_dir, name="predictions", exist_ok=True)

    # Display results
    for result in results:
        # result.plot() gives a numpy array with the drawn predictions
        im = result.plot()
        # cv2.imshow("Prediction", im) - the VM we ran on doesn't have needed drivers
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO pose prediction on an image.")
    parser.add_argument("--weights", type=str, default="part3_best.pt", help="best.pt")
    parser.add_argument("--source", type=str, default="detections/images/frame_0001729.jpg", help="Path to input image")
    parser.add_argument("--save_dir", type=str, default="final_model_eval", help="Directory to save predictions")

    args = parser.parse_args()
    run_prediction(args.weights, args.source, args.save_dir)
