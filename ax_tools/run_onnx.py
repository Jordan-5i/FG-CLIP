import numpy as np
from PIL import Image
import onnxruntime as ort
import torch
import os
import tarfile
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


def prepare_calib():
    calib_dir = "calib_data"
    if not os.path.exists(calib_dir):
        os.makedirs(calib_dir)
    image_sub_dir = os.path.join(calib_dir, "image")
    text_sub_dir = os.path.join(calib_dir, "text")
    if not os.path.exists(image_sub_dir):
        os.makedirs(image_sub_dir)
    if not os.path.exists(text_sub_dir):
        os.makedirs(text_sub_dir)
        
    image_tar = tarfile.open(os.path.join(image_sub_dir, "calib_data_image.tar"), "w")
    text_tar = tarfile.open(os.path.join(text_sub_dir, "calib_data_text.tar"), "w")
    
    np.save(os.path.join(image_sub_dir, "image_input.npy"), 
            {"pixel_values": image_input["pixel_values"].numpy().astype(np.float32), 
            "pixel_attention_mask": image_input["pixel_attention_mask"].numpy().astype(np.int32),
            "spatial_shapes": image_input["spatial_shapes"].numpy().astype(np.int64)})
    image_tar.add(os.path.join(image_sub_dir, "image_input.npy"))
    
    for i, c in enumerate(caption_input["input_ids"]):
        np.save(os.path.join(text_sub_dir, f"input_ids_{i}.npy"), {"input_ids": c[None].numpy().astype(np.int64)})
        text_tar.add(os.path.join(text_sub_dir, f"input_ids_{i}.npy"))
    
    text_tar.close()
    image_tar.close()
    
    return
    
if __name__ == "__main__":
    image_path = "bedroom.jpg"
    model_root = "/data/wangjian/project/hf_cache/qihoo360/fg-clip2-base"
    
    image_encoder_path = "image_encoder.onnx"
    text_encoder_path = "text_encoder.onnx"
    
    onnx_image_encoder = ort.InferenceSession(image_encoder_path)
    onnx_text_encoder = ort.InferenceSession(text_encoder_path)
    
    image = Image.open(image_path).convert("RGB")
    
    image_processor = AutoImageProcessor.from_pretrained(model_root)
    tokenizer = AutoTokenizer.from_pretrained(model_root)
    
    image_input = image_processor(images=image, max_num_patches=determine_max_value(image), return_tensors="pt")
    captions = [
        "一个简约风格的卧室角落，黑色金属衣架上挂着多件米色和白色的衣物，下方架子放着两双浅色鞋子，旁边是一盆绿植，左侧可见一张铺有白色床单和灰色枕头的床。",
        "一个简约风格的卧室角落，黑色金属衣架上挂着多件红色和蓝色的衣物，下方架子放着两双黑色高跟鞋，旁边是一盆绿植，左侧可见一张铺有白色床单和灰色枕头的床。",
        "一个简约风格的卧室角落，黑色金属衣架上挂着多件米色和白色的衣物，下方架子放着两双运动鞋，旁边是一盆仙人掌，左侧可见一张铺有白色床单和灰色枕头的床。",
        "一个繁忙的街头市场，摊位上摆满水果，背景是高楼大厦，人们在喧闹中购物。"
    ]
    captions = [caption.lower() for caption in captions]

    caption_input = tokenizer(captions, padding="max_length", max_length=196, truncation=True, return_tensors="pt")

    image_feature = onnx_image_encoder.run(None, {
        "pixel_values": image_input["pixel_values"].numpy().astype(np.float32),
        "pixel_attention_mask": image_input["pixel_attention_mask"].numpy().astype(np.int32),
        "spatial_shapes": image_input["spatial_shapes"].numpy().astype(np.int64)
    })[0]

    text_feature = []
    for c in caption_input["input_ids"]:
        tmp_text_feature = onnx_text_encoder.run(None, {
            "input_ids": c[None].numpy().astype(np.int64),
        })[0]
        text_feature.append(tmp_text_feature)
    text_feature = np.concatenate(text_feature, axis=0)
    
    logits_per_image = image_feature @ text_feature.T
    logit_scale, logit_bias = 4.75, -16.75
    logits_per_image = logits_per_image * np.exp(logit_scale) + logit_bias

    print("Logits per image:", torch.from_numpy(logits_per_image).softmax(dim=-1))
    
    prepare_calib()