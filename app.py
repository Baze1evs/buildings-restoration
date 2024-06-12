import gradio as gr
import torch
from torchvision.transforms import v2
from transformers import AutoModel
from networks import Generator, Discriminator, DINOWrapper, TravelGan


def eval(img):
    test_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    img = test_transform(img).reshape(-1, 3, 224, 224).to(model.device)
    model.gen.eval()
    pred = model.gen(img)[0]
    return v2.functional.to_pil_image(pred*0.5+0.5)

backbone = AutoModel.from_pretrained('facebook/dinov2-base')
for param in backbone.parameters():
    param.requires_grad = False

gen = Generator(3, num_feat=64, num_res=5)
dis = Discriminator(3)
siam = DINOWrapper(backbone)

model = TravelGan(gen, dis, siam)

gr.Interface(fn=eval,
             inputs=gr.Image(type="pil"),
             outputs="image",
             examples=["celeba_hq/val/male/000406.jpg"]).launch(share=True, mirror_camera=False)
