import os
import skimage
import torch, os
from PIL import Image
from pathlib import Path
from briarmbg import BriaRMBG
from huggingface_hub import hf_hub_download
from utilities import preprocess_image, postprocess_image


def infer(net, orig_im, model_input_size, device, orig_im_size):
    image = preprocess_image(orig_im, model_input_size).to(device)
    result = net(image)
    result_image = postprocess_image(result[0][0], orig_im_size)
    return result_image


def example_inference(net, device, im_path, out_path):
    # prepare input
    model_input_size = [1024, 1024]
    orig_im = skimage.io.imread(im_path)
    orig_im_size = orig_im.shape[0:2]

    result_image = infer(net, orig_im, model_input_size, device, orig_im_size)

    # save result
    pil_im = Image.fromarray(result_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    orig_image = Image.open(im_path)
    no_bg_image.paste(orig_image, mask=pil_im)
    no_bg_image.save(out_path)


def main():
    net = BriaRMBG()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = hf_hub_download("briaai/RMBG-1.4", "model.pth")
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    image_directory = "images/"
    outs_path = "outputs/RMBG-1.4/"
    os.makedirs(outs_path, exist_ok=True)
    for filename in os.listdir(image_directory):
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff")
        ):

            im_path = image_directory + filename
            out_path = outs_path + filename
            out_path = Path(out_path)
            out_path = out_path.with_suffix(".png")

            example_inference(net, device, im_path, out_path)


if __name__ == "__main__":
    main()
