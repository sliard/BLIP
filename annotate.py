from PIL import Image
import argparse
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder

device = torch.device('cpu')

def load_demo_image(image_path,image_size,device):
    raw_image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


parser = argparse.ArgumentParser()
parser.add_argument('--image')
parser.add_argument('--model', default='checkpoints/model_base_caption_capfilt_large.pth')
parser.add_argument('--config', default='configs/med_config.json')
args = parser.parse_args()

image_size = 384
image = load_demo_image(image_path=args.image, image_size=image_size, device=device)

model = blip_decoder(pretrained=args.model, image_size=image_size, vit='base', med_config=args.config)
model.eval()
model = model.to(device)

with torch.no_grad():
    # beam search
    caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
    # nucleus sampling
    # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
    print('**caption:'+caption[0]+'**')

