import os
import cv2
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset


class CXRDiffusionDataset(Dataset):

    def __init__(
        self,
        csv_path,
        base_image_dir="",
        resolution=512
    ):
        self.df = pd.read_csv(csv_path)

        self.base_image_dir = base_image_dir
        self.resolution = resolution

        # Classes present in the diffusion training dataset
        self.target_classes = [
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Lung Lesion",
            "Pneumonia",
            "Pleural Other",
            "Fracture"
        ]

        # Check required columns
        missing_columns = [
            col
            for col in self.target_classes
            if col not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns in CSV: {missing_columns}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # ============================================================
        # 1. LOAD IMAGE
        # ============================================================

        img_path = os.path.join(
            self.base_image_dir,
            row["Path"]
        )

        image = cv2.imread(
            img_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise FileNotFoundError(
                f"Image not found: {img_path}"
            )

        # ============================================================
        # 2. DIRECT RESIZE
        # ============================================================

        image = cv2.resize(
            image,
            (self.resolution, self.resolution),
            interpolation=cv2.INTER_CUBIC
        )

        # ============================================================
        # 3. GRAYSCALE -> RGB
        # ============================================================

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

        # ============================================================
        # 4. NORMALIZE
        #
        # [0, 255] -> [-1, 1]
        # ============================================================

        image = image.astype(np.float32)
        image = image / 127.5 - 1.0

        # ============================================================
        # 5. HWC -> CHW
        #
        # OpenCV: H x W x C
        # PyTorch: C x H x W
        # ============================================================

        pixel_values = torch.from_numpy(
            image
        ).permute(2, 0, 1)

        # ============================================================
        # 6. FIND THE ACTIVE LABEL
        # ============================================================

        findings = [
            col
            for col in self.target_classes
            if row[col] == 1.0
        ]

        # Dataset should already be single-label
        if len(findings) != 1:
            raise ValueError(
                f"Expected exactly one label at row {idx}, "
                f"but found: {findings}"
            )

        # ============================================================
        # 7. CREATE TEXT CAPTION
        # ============================================================

        if findings[0] == "No Finding":
            caption = (
                "Normal chest X-ray with no significant findings"
            )
        else:
            caption = (
                "Chest X-ray showing "
                + findings[0]
            )

        # ============================================================
        # 8. RETURN
        # ============================================================

        return {
            "pixel_values": pixel_values,
            "prompt": caption
        }


def collate_fn(examples):

    pixel_values = torch.stack(
        [
            example["pixel_values"]
            for example in examples
        ]
    )

    prompts = [
        example["prompt"]
        for example in examples
    ]

    return {
        "pixel_values": pixel_values,
        "prompts": prompts
    }