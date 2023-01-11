from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_demo_image(image_size,device):
    img_url = 'https://controller-stadiffdev-1.babbar.eu/pictures/bf82a566-5dc8-4ac1-9600-d835e7ebcc76/d3b514b7-8573-4e59-bca9-9b530a81caeb.jpg'
#    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    raw_image = Image.open('/home/sliard/dev/yourtextguru/ytg-images/controller/src/main/resources/static/pictures/8896a15c-0bed-48be-afa0-69ab21e3dc27/0147486d-61ce-4ca7-8c9c-b8b10a133beb.jpg').convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


image_size = 384
image = load_demo_image(image_size=image_size, device=device)

model_url = '/home/sliard/dev/proto/BLIP/checkpoints/model_base_caption_capfilt_large.pth'

model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

with torch.no_grad():
    # beam search
    caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
    # nucleus sampling
    # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
    print('caption: '+caption[0])