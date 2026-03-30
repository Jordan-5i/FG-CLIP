import torch
import types
from torch import nn
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from demo import determine_max_value


def new_resize_positional_embeddings(self, positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        max_length: int,
    ) -> torch.Tensor:

        """
        Resize positional embeddings to image-specific size and pad to a fixed size.

        Args:
            positional_embeddings (`torch.Tensor`):
                Position embeddings of shape (height, width, embed_dim)
            spatial_shapes (`torch.LongTensor`):
                Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
            max_length (`int`):
                Maximum length of the positional embeddings to pad resized positional embeddings to

        Returns:
            `torch.Tensor`: Embeddings of shape (batch_size, max_length, embed_dim)
        """
        spatial_shapes = torch.tensor([[30, 34]])
        batch_size = spatial_shapes.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        resulted_positional_embeddings = torch.empty(
            (batch_size, max_length, embed_dim),
            device=positional_embeddings.device,
            dtype=source_dtype,
        )

        # (height, width, embed_dim) -> (1, embed_dim, height, width) for interpolation
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # Upcast to float32 on CPU because antialias is not supported for bfloat16/float16 on CPU
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.to(torch.float32)

        for i in range(batch_size):
            # (1, dim, height, width) -> (1, dim, target_height, target_width)
            height, width = spatial_shapes[i]
            resized_embeddings = F.interpolate(
                positional_embeddings,
                size=(height, width),
                mode="nearest",  # PS: 原始代码是 bilinear，onnx不支持 ):
            )

            # (1, dim, target_height, target_width) -> (target_height * target_width, dim)
            resized_embeddings = resized_embeddings.reshape(embed_dim, height * width).transpose(0, 1)

            # Cast to original dtype
            resized_embeddings = resized_embeddings.to(source_dtype)

            resulted_positional_embeddings[i, : height * width] = resized_embeddings
            resulted_positional_embeddings[i, height * width :] = resized_embeddings[0]

        return resulted_positional_embeddings


class FGClipImageEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, pixel_values) -> torch.Tensor:
        image_feature = self.model.get_image_features(pixel_values=pixel_values)
        image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)
        return image_feature
    
    
class FGClip2ImageEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, pixel_values, pixel_attention_mask, spatial_shapes) -> torch.Tensor:
        image_feature = self.model.get_image_features(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes)
        image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)
        return image_feature


class FGClipTextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask=None) -> torch.Tensor:
        text_feature = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)
        return text_feature
    

class FGClip2TextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask=None, walk_type="long") -> torch.Tensor:
        text_feature = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask, walk_type=walk_type)
        text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)
        return text_feature



if __name__ == "__main__":
    model_root = "/data/wangjian/project/hf_cache/qihoo360/fg-clip-base"
    model = AutoModelForCausalLM.from_pretrained(model_root, trust_remote_code=True)
    device = model.device

    tokenizer = AutoTokenizer.from_pretrained(model_root)
    image_processor = AutoImageProcessor.from_pretrained(model_root)
    
    img_root = "bedroom.jpg"
    image = Image.open(img_root).convert("RGB")

    image_input = image_processor(images=image, max_num_patches=determine_max_value(image), return_tensors="pt").to(device)
    
    captions = [
        "一个简约风格的卧室角落，黑色金属衣架上挂着多件米色和白色的衣物，下方架子放着两双浅色鞋子，旁边是一盆绿植，左侧可见一张铺有白色床单和灰色枕头的床。",
        # "一个简约风格的卧室角落，黑色金属衣架上挂着多件红色和蓝色的衣物，下方架子放着两双黑色高跟鞋，旁边是一盆绿植，左侧可见一张铺有白色床单和灰色枕头的床。",
        # "一个简约风格的卧室角落，黑色金属衣架上挂着多件米色和白色的衣物，下方架子放着两双运动鞋，旁边是一盆仙人掌，左侧可见一张铺有白色床单和灰色枕头的床。",
        # "一个繁忙的街头市场，摊位上摆满水果，背景是高楼大厦，人们在喧闹中购物。"
    ]
    captions = [caption.lower() for caption in captions]

    caption_input = tokenizer(captions, padding="max_length", max_length=196, truncation=True, return_tensors="pt").to(device)
    
    
    Fgclip2VisionEmbeddings = model.vision_model.embeddings.__class__
    for m in model.modules():
        if isinstance(m, Fgclip2VisionEmbeddings):
            m.resize_positional_embeddings = types.MethodType(new_resize_positional_embeddings, m)

    # v1
    if "fg-clip-base" in model_root:
        # export image onnx
        image_encoder = FGClipImageEncoder(model)
        torch.onnx.export(image_encoder,
            (image_input['pixel_values']),
            f"image_encoder.onnx",
            input_names=['pixel_values'],
            output_names=['norm_image_features'],
            export_params=True,
            opset_version=17,)
        
        # export text onnx
        text_encoder = FGClipTextEncoder(model)
        torch.onnx.export(text_encoder,
                    (caption_input['input_ids'], ),
                    f"text_encoder.onnx",
                    input_names=['input_ids', ],
                    output_names=['norm_text_features'],
                    export_params=True,
                    opset_version=17,)
    else: # v2
        # export image onnx
        image_encoder = FGClip2ImageEncoder(model)
        torch.onnx.export(image_encoder,
                    (image_input['pixel_values'], image_input['pixel_attention_mask'], image_input['spatial_shapes']),
                    f"image_encoder.onnx",
                    input_names=['pixel_values', 'pixel_attention_mask', 'spatial_shapes'],
                    output_names=['norm_image_features'],
                    export_params=True,
                    opset_version=17,)
        
        # export text onnx
        text_encoder = FGClip2TextEncoder(model)
        torch.onnx.export(text_encoder,
                    (caption_input['input_ids'], ),
                    f"text_encoder.onnx",
                    input_names=['input_ids', ],
                    output_names=['norm_text_features'],
                    export_params=True,
                    opset_version=17,)
