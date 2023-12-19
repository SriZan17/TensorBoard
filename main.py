import torchvision
from going_modular import helper_functions
from going_modular import data_setup
from going_modular.engine import train
from going_modular.helper_functions import create_writer
import torch
from torch import nn
from going_modular.helper_functions import download_data
from going_modular.utils import save_model


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    # image_path = helper_functions.download_data(
    #    source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    #    destination="pizza_steak_sushi",
    # )
    data_10_percent_path = download_data(
        source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
        destination="pizza_steak_sushi",
    )
    data_20_percent_path = download_data(
        source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
        destination="pizza_steak_sushi_20_percent",
    )

    # train_dir = image_path / "train"
    # test_dir = image_path / "test"
    train_dir_10_percent = data_10_percent_path / "train"
    train_dir_20_percent = data_20_percent_path / "train"

    test_dir = data_10_percent_path / "test"

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    automatic_transforms = weights.transforms()

    # train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    #    train_dir=train_dir,
    #    test_dir=test_dir,
    #    transform=automatic_transforms,
    #    batch_size=32,
    # )

    (
        train_dataloader_10_percent,
        test_dataloader,
        class_names,
    ) = data_setup.create_dataloaders(
        train_dir=train_dir_10_percent,
        test_dir=test_dir,
        transform=automatic_transforms,
        batch_size=BATCH_SIZE,
    )
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

    # model = torchvision.models.efficientnet_b0(weights=weights).to(DEVICE)
    # helper_functions.set_seeds(43)
    # for param in model.features.parameters():
    #    param.requires_grad = False
    # model.classifier = torch.nn.Sequential(
    #    nn.Dropout(p=0.2, inplace=True),
    #    nn.Linear(
    #        in_features=1280,
    #        out_features=len(class_names),
    #        bias=True,
    #    ).to(DEVICE),
    # )

    # 1. Create epochs list
    num_epochs = [5, 10]

    # 2. Create models list (need to create a new model for each experiment)
    models = ["effnetb0", "effnetb2"]

    # 3. Create dataloaders dictionary for various dataloaders
    train_dataloaders = {
        "data_10_percent": train_dataloader_10_percent,
        "data_20_percent": train_dataloader_20_percent,
    }

    # loss_function = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # writer = create_writer(
    #    experiment_name="efficientnet",
    #    model_name="efficientnet_b0",
    #    extra=str(NUM_EPOCHS) + "_epochs",
    # )
    # results = train(
    #    model=model,
    #    train_dataloader=train_dataloader,
    #    test_dataloader=test_dataloader,
    #    optimizer=optimizer,
    #    loss_fn=loss_function,
    #    epochs=NUM_EPOCHS,
    #    device=DEVICE,
    #    writer=writer,
    # )
    # print(results)

    # 1. Set the random seeds
    helper_functions.set_seeds(seed=43)

    # 2. Keep track of experiment numbers
    experiment_number = 0

    # 3. Loop through each DataLoader
    for dataloader_name, train_dataloader in train_dataloaders.items():
        # 4. Loop through each number of epochs
        for epochs in num_epochs:
            # 5. Loop through each model name and create a new model based on the name
            for model_name in models:
                # 6. Create information print outs
                experiment_number += 1
                print(f"[INFO] Experiment number: {experiment_number}")
                print(f"[INFO] Model: {model_name}")
                print(f"[INFO] DataLoader: {dataloader_name}")
                print(f"[INFO] Number of epochs: {epochs}")

                # 7. Select the model
                if model_name == "effnetb0":
                    model = create_effnetb0(
                        device=DEVICE, OUT_FEATURES=len(class_names)
                    )  # creates a new model each time (important because we want each experiment to start from scratch)
                else:
                    model = create_effnetb2(
                        device=DEVICE, OUT_FEATURES=len(class_names)
                    )  # creates a new model each time (important because we want each experiment to start from scratch)

                # 8. Create a new loss and optimizer for every model
                loss_fn = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

                # 9. Train target model with target dataloaders and track experiments
                train(
                    model=model,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    epochs=epochs,
                    device=DEVICE,
                    writer=create_writer(
                        experiment_name=dataloader_name,
                        model_name=model_name,
                        extra=f"{epochs}_epochs",
                    ),
                )

                # 10. Save the model to file so we can get back the best model
                save_filepath = f"07_{model_name}_{dataloader_name}_{epochs}_epochs.pth"
                save_model(model=model, target_dir="models", model_name=save_filepath)
                print("-" * 50 + "\n")


def create_effnetb0(device, OUT_FEATURES):
    # 1. Get the base mdoel with pretrained weights and send to target device
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    # 2. Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Set the seeds
    helper_functions.set_seeds()

    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2), nn.Linear(in_features=1280, out_features=OUT_FEATURES)
    ).to(device)

    # 5. Give the model a name
    model.name = "effnetb0"
    print(f"[INFO] Created new {model.name} model.")
    return model


def create_effnetb2(device, OUT_FEATURES):
    # 1. Get the base model with pretrained weights and send to target device
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights).to(device)

    # 2. Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Set the seeds
    helper_functions.set_seeds()

    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(in_features=1408, out_features=OUT_FEATURES)
    ).to(device)

    # 5. Give the model a name
    model.name = "effnetb2"
    print(f"[INFO] Created new {model.name} model.")
    return model


if __name__ == "__main__":
    main()
