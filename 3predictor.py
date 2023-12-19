import torch
from main import create_effnetb2
from going_modular.predictions import pred_and_plot_image
import random
from pathlib import Path
from going_modular.helper_functions import download_data
from going_modular import data_setup
import torchvision

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

data_20_percent_path = download_data(
    source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
    destination="pizza_steak_sushi_20_percent",
)
train_dir_20_percent = data_20_percent_path / "train"
weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
automatic_transforms = weights.transforms()

test_dir = data_20_percent_path / "test"
(
    train_dataloader_20_percent,
    test_dataloader,
    class_names,
) = data_setup.create_dataloaders(
    train_dir=train_dir_20_percent,
    test_dir=test_dir,
    transform=automatic_transforms,
    batch_size=BATCH_SIZE,
)

# Setup the best model filepath
best_model_path = "models/07_effnetb2_data_20_percent_10_epochs.pth"

# Instantiate a new instance of EffNetB2 (to load the saved state_dict() to)
best_model = create_effnetb2(device=DEVICE, OUT_FEATURES=len(class_names))

# Load the saved best model state_dict()
best_model.load_state_dict(torch.load(best_model_path))
num_images_to_plot = 3
test_image_path_list = list(
    Path(data_20_percent_path / "test").glob("*/*.jpg")
)  # get all test image paths from 20% dataset
test_image_path_sample = random.sample(
    population=test_image_path_list, k=num_images_to_plot
)  # randomly select k number of images

# Iterate through random test image paths, make predictions on them and plot them
for image_path in test_image_path_sample:
    pred_and_plot_image(
        model=best_model,
        image_path=image_path,
        class_names=class_names,
        image_size=(224, 224),
    )
