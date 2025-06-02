import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(size):
    train_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                # Color and lighting
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2),
                A.ToGray(p=0.1),

                # Blur and noise
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.RandomGamma(gamma_limit=(90, 110), p=0.3),

                # Distortions
                A.ElasticTransform(alpha=1, sigma=50, p=0.2),
                A.GridDistortion(p=0.2),
                A.OpticalDistortion(distort_limit=0.05, p=0.2),


                # Final conversion
                A.Resize(height = size, width = size, p=1.0),
                A.Normalize(mean=(0.485, 0.456, 0.406),  # For ImageNet pretrained models
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),

    ])

    valid_transform = A.Compose([
                A.Resize(height = size, width = size, p=1.0),
                A.Normalize(p = 1.0),
                ToTensorV2()
    ])
    return train_transform, valid_transform
