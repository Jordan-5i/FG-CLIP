import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)


def determine_max_value(image):
    w,h = image.size
    max_val = (w//16)*(h//16)
    if max_val > 784:
        return 1024
    elif max_val > 576:
        return 784
    elif max_val > 256:
        return 576
    elif max_val > 128:
        return 256
    else:
        return 128
    
    
if __name__ == "__main__":

    model_root = "/data/wangjian/project/hf_cache/qihoo360/fg-clip2-base"
    model = AutoModelForCausalLM.from_pretrained(model_root,trust_remote_code=True).cuda(1)

    device = model.device

    tokenizer = AutoTokenizer.from_pretrained(model_root)
    image_processor = AutoImageProcessor.from_pretrained(model_root)

    img_root = "bedroom.jpg"
    image = Image.open(img_root).convert("RGB")

    image_input = image_processor(images=image, max_num_patches=determine_max_value(image), return_tensors="pt").to(device)

    # NOTE Short captions: max_length=64 walk_type="short"(default)
    # NOTE Long captions: max_length=196 walk_type="long"

    captions = [
        "一个简约风格的卧室角落，黑色金属衣架上挂着多件米色和白色的衣物，下方架子放着两双浅色鞋子，旁边是一盆绿植，左侧可见一张铺有白色床单和灰色枕头的床。",
        "一个简约风格的卧室角落，黑色金属衣架上挂着多件红色和蓝色的衣物，下方架子放着两双黑色高跟鞋，旁边是一盆绿植，左侧可见一张铺有白色床单和灰色枕头的床。",
        "一个简约风格的卧室角落，黑色金属衣架上挂着多件米色和白色的衣物，下方架子放着两双运动鞋，旁边是一盆仙人掌，左侧可见一张铺有白色床单和灰色枕头的床。",
        "一个繁忙的街头市场，摊位上摆满水果，背景是高楼大厦，人们在喧闹中购物。"
    ]
    captions = [caption.lower() for caption in captions]

    caption_input = tokenizer(captions, padding="max_length", max_length=196, truncation=True, return_tensors="pt").to(device)


    with torch.no_grad():
        image_feature = model.get_image_features(**image_input)
        text_feature = model.get_text_features(**caption_input,walk_type="long")
        image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)
        text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)

    logits_per_image = image_feature @ text_feature.T
    logit_scale, logit_bias = model.logit_scale.to(text_feature.device), model.logit_bias.to(text_feature.device)
    logits_per_image = logits_per_image * logit_scale.exp() + logit_bias

    print("Logits per image:", logits_per_image.softmax(dim=-1))