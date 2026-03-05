import yaml
import zipfile
import os
import mlflow
from ultralytics import YOLO, settings

#Configuration 
NUM_EPOCHS   = 60
BATCH_SIZE   = 8
IMG_SIZE     = 640
DATASET_YAML = '/content/dataset.yaml'
SAVE_PATH    = "/content/drive/MyDrive/SUMO Grad Proj Output files/yolo_results"

# Setup Dataset
def setup_dataset():
    zip_path     = '/content/drive/MyDrive/SUMO Grad Proj Output files/dataset_v2.zip'
    extract_path = '/content/dataset_v2'

    if not os.path.exists(extract_path):
        print("Unzipping dataset")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Dataset unzipped!")
    else:
        print("Dataset already exists!")

    dataset_config = {
        'path': '/content/dataset_v2',
        'train': 'images/train',
        'val':   'images/val',
        'test':  'images/test',
        'nc': 7,
        'names': ['car', 'bus', 'truck', 'motorcycle',
                  'taxi', 'microbus', 'bicycle']
    }

    with open('/content/dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f)

    print("dataset.yaml recreated!")
    print(f"   Train: {len(os.listdir('/content/dataset_v2/images/train'))} images")
    print(f"   Val:   {len(os.listdir('/content/dataset_v2/images/val'))} images")
    print(f"   Test:  {len(os.listdir('/content/dataset_v2/images/test'))} images")

# Training 
def main():
    setup_dataset()

    # Disable YOLO built-in MLflow
    settings.update({"mlflow": False})

    # Setup MLflow
    mlflow.end_run()
    mlflow.set_tracking_uri(
        "sqlite:////content/drive/MyDrive/SUMO Grad Proj Output files/mlflow.db"
    )
    mlflow.set_experiment("SumoFlowAI-Traffic-Detection")

    print("Loading pre-trained YOLOv8s")
    model = YOLO("yolov8s.pt")
    print("Training on device: GPU (CUDA)")

    with mlflow.start_run(run_name="yolo_v3_small_model"):
        mlflow.log_param("num_epochs",  NUM_EPOCHS)
        mlflow.log_param("batch_size",  BATCH_SIZE)
        mlflow.log_param("img_size",    IMG_SIZE)
        mlflow.log_param("optimizer",   "AdamW")
        mlflow.log_param("model",       "yolov8s")

        results = model.train(
            data       = DATASET_YAML,
            epochs     = NUM_EPOCHS,
            imgsz      = IMG_SIZE,
            batch      = BATCH_SIZE,
            cls        = 2.0,
            degrees    = 10.0,
            mixup      = 0.1,
            copy_paste = 0.1,
            name       = "tahrir_yolo_v3",
            project    = SAVE_PATH,
            save       = True,
            plots      = True,
            device     = 0,
            verbose    = True,
        )

        mAP50     = results.results_dict['metrics/mAP50(B)']
        mAP50_95  = results.results_dict['metrics/mAP50-95(B)']
        precision = results.results_dict['metrics/precision(B)']
        recall    = results.results_dict['metrics/recall(B)']

        mlflow.log_metric("mAP50",     mAP50)
        mlflow.log_metric("mAP50_95",  mAP50_95)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)

        best_weights = f"{SAVE_PATH}/tahrir_yolo_v3/weights/best.pt"
        if os.path.exists(best_weights):
            mlflow.log_artifact(best_weights)

        print("        YOLO FINAL RESULTS")
        print(f"  mAP@0.5    : {mAP50:.4f}  ({mAP50*100:.2f}%)")
        print(f"  mAP@0.5:95 : {mAP50_95:.4f}  ({mAP50_95*100:.2f}%)")
        print(f"  Precision  : {precision:.4f}  ({precision*100:.2f}%)")
        print(f"  Recall     : {recall:.4f}  ({recall*100:.2f}%)")


    print(" Training Complete!")

if __name__ == "__main__":
    main()