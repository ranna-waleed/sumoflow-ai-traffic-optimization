import yaml
import zipfile
import os
import mlflow
from ultralytics import YOLO, settings
import torch

#Configuration 
NUM_EPOCHS   = 60
BATCH_SIZE   = 8
IMG_SIZE     = 640
DATASET_YAML = 'detection/dataset/dataset.yaml'
SAVE_PATH    = 'detection/yolo_results'

# Setup Dataset
def setup_dataset():
    dataset_config = {
        'path': os.path.abspath('detection/dataset'),
        'train': 'images/train',
        'val':   'images/val',
        'test':  'images/test',
        'nc': 7,
        'names': ['car', 'bus', 'truck', 'motorcycle',
                  'taxi', 'microbus', 'bicycle']
    }

    yaml_path = 'detection/dataset/dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f)

    print("dataset.yaml created!")
    print(f"   Train: {len(os.listdir('detection/dataset/images/train'))} images")
    print(f"   Val:   {len(os.listdir('detection/dataset/images/val'))} images")
    print(f"   Test:  {len(os.listdir('detection/dataset/images/test'))} images")

# Training 
def main():
    setup_dataset()

    # Disable YOLO built-in MLflow
    settings.update({"mlflow": False})

    # Setup MLflow
    mlflow.end_run()
    mlflow.set_tracking_uri(
        f"sqlite:///{os.path.abspath('mlflow.db')}"
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
            cls        = 2.0, # multiplies the classification loss by 2, making the model prioritize getting the class right over just finding the box
            degrees    = 10.0, #randomly rotates images ±10° during training for augmentation.
            mixup      = 0.1, # 10% of batches blend two images together as augmentation, forcing the model to handle overlapping objects.
            copy_paste = 0.1, # randomly copies objects from one image and pastes them into another, for rare classes like bicycles/motorcycles
            name       = "tahrir_yolo_v3",
            project    = SAVE_PATH,
            save       = True,
            plots      = True,
            device = 0 if torch.cuda.is_available() else 'cpu',
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