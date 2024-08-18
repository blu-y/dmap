import torch
from open_clip import create_model_from_pretrained, get_tokenizer

model_name = 'ViT-B-16-SigLIP'
checkpoint_path = 'models/ViT-B-16-SigLIP/open_clip_pytorch_model.bin'
device = torch.device('cpu')

model, _ = create_model_from_pretrained(model_name, checkpoint_path, device)
tokenizer = get_tokenizer(model_name)

text = tokenizer(["a photo of a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.tolist()
    text_features = text_features[0]

print(text_features)