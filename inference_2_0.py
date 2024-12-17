import os
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


def infer(model, input_images):
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    return preds


def example_inference(model, device, im_path, out_path):
    image_size = (1024, 1024)
    transform_image = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(im_path)
    input_images = transform_image(image).unsqueeze(0).to(device)

    # Prediction
    preds = infer(model, input_images)

    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    image.save(out_path)


def main():
    model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-2.0", trust_remote_code=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision(["high", "highest"][0])
    model.to(device)
    model.eval()

    image_directory = "images/"
    outs_path = "outputs/RMBG-2.0/"
    os.makedirs(outs_path, exist_ok=True)
    for filename in os.listdir(image_directory):
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff")
        ):

            im_path = image_directory + filename
            out_path = outs_path + filename
            out_path = Path(out_path)
            out_path = out_path.with_suffix(".png")

            example_inference(model, device, im_path, out_path)


if __name__ == "__main__":
    main()
