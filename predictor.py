# Download custom image
import requests
import torch
from pathlib import Path
from going_modular.predictions import pred_and_plot_image
from main import create_effnetb2

# Setup custom image path
custom_image_path = Path("data/04-pizza-dad.jpeg")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Download the image if it doesn't already exist
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        # When downloading from GitHub, need to use the "raw" file link
        request = requests.get(
            "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg"
        )
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download.")

class_names = ["Pizza", "Steak", "Sushi"]
model = create_effnetb2(device=DEVICE, OUT_FEATURES=len(class_names))
model = torch.compile(model)
model.load_state_dict(torch.load("models/07_effnetb2_data_20_percent_10_epochs.pth"))


# Predict on custom image
pred_and_plot_image(model=model, image_path=custom_image_path, class_names=class_names)
