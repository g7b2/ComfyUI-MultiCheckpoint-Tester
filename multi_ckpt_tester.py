import os
import torch
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json
import gc  # 引入垃圾回收模块
import traceback

import comfy.sd
import comfy.utils
import nodes
import folder_paths
import comfy.model_management

class MultiCheckpointIncrementalNamer:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        checkpoints = folder_paths.get_filename_list("checkpoints")
        return {
            "required": {
                "ckpt_name_1": (checkpoints,),
                "positive_prompt": ("STRING", {"multiline": True, "default": "1girl, cinematic lighting"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "low quality, blurry"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 0.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "width": ("INT", {"default": 512, "step": 8}),
                "height": ("INT", {"default": 512, "step": 8}),
            },
            "optional": {
                "ckpt_name_2": (["None"] + checkpoints,),
                "ckpt_name_3": (["None"] + checkpoints,),
                "ckpt_name_4": (["None"] + checkpoints,),
                "ckpt_name_5": (["None"] + checkpoints,),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_test"
    CATEGORY = "CustomNodes/Test"
    OUTPUT_NODE = True

    def get_unique_path(self, base_name):
        filename = f"{base_name}.png"
        full_path = os.path.join(self.output_dir, filename)
        if os.path.exists(full_path):
            counter = 1
            while True:
                suffix = f"_{counter:02d}"
                filename = f"{base_name}{suffix}.png"
                full_path = os.path.join(self.output_dir, filename)
                if not os.path.exists(full_path):
                    break
                counter += 1
        return full_path

    def run_test(self, ckpt_name_1, positive_prompt, negative_prompt, seed, steps, cfg, sampler_name, scheduler, denoise, width, height, prompt=None, extra_pnginfo=None, **kwargs):
        # 收集所有非空的模型名称
        selected_ckpts = [ckpt_name_1]
        for i in range(2, 6):
            name = kwargs.get(f"ckpt_name_{i}")
            if name and name != "None":
                selected_ckpts.append(name)

        final_images_list = []
        
        # 准备元数据
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        for full_name in selected_ckpts:
            clean_name = os.path.splitext(os.path.basename(full_name))[0]
            print(f"🔄 [Multi-Ckpt] 正在处理模型: {clean_name}")

            # 初始化变量防止 finally 中引用未定义变量
            model = clip = vae = sample = None
            
            try:
                # 1. 加载模型
                ckpt_path = folder_paths.get_full_path("checkpoints", full_name)
                # 使用 comfy 的加载器
                out = comfy.sd.load_checkpoint_guess_config(
                    ckpt_path, 
                    output_vae=True, 
                    output_clip=True, 
                    embedding_directory=folder_paths.get_folder_paths("embeddings")
                )
                model, clip, vae = out[0], out[1], out[2]

                # 2. 编码 Prompt
                tokens_pos = clip.tokenize(positive_prompt)
                cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
                positive = [[cond_pos, {"pooled_output": pooled_pos}]]

                tokens_neg = clip.tokenize(negative_prompt)
                cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
                negative = [[cond_neg, {"pooled_output": pooled_neg}]]

                # 3. 采样 (KSampler)
                latent = torch.zeros([1, 4, height // 8, width // 8], device=comfy.model_management.get_torch_device())
                samples = {"samples": latent}
                
                # 执行采样
                sample = nodes.common_ksampler(
                    model, seed, steps, cfg, sampler_name, scheduler, 
                    positive, negative, samples, denoise=denoise
                )[0]

                # 4. 解码 (VAE Decode) -> 输出通常是 [1, C, H, W]
                # 注意：VAE 解码需要在 GPU 上进行以提高速度，但要小心显存
                decoded_tensor = vae.decode(sample["samples"])

                # --- 关键修复 1: 维度转换 ---
                # 从 [Batch, Channel, Height, Width] 转换为 [Batch, Height, Width, Channel]
                # 这是 ComfyUI 图像管道的标准格式
                if decoded_tensor.shape[1] == 3: # 确保是 C 在第二维
                    decoded_tensor = decoded_tensor.permute(0, 2, 3, 1)
                
                # 将处理好的 Tensor 加入列表用于最后返回
                # 将 Tensor 移回 CPU 以节省显存，防止在列表中堆积占用 GPU
                final_images_list.append(decoded_tensor.cpu())

                # 5. 保存图片
                save_path = self.get_unique_path(clean_name)
                
                # 转换用于 PIL 保存: 
                # tensor [1, H, W, C] -> squeeze -> [H, W, C] -> numpy
                img_array = 255. * decoded_tensor.cpu().numpy().squeeze()
                img_pil = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
                
                img_pil.save(save_path, pnginfo=metadata, compress_level=4)
                print(f"✅ [Multi-Ckpt] 成功保存: {save_path}")

            except Exception as e:
                # --- 关键修复 3: 异常捕获 ---
                print(f"❌ [Multi-Ckpt] 处理模型 {clean_name} 时发生错误:\n{traceback.format_exc()}")
                # 可以选择添加一个全黑图片占位，或者直接跳过
                continue

            finally:
                # --- 关键修复 2: 暴力显存清理 ---
                # 显式删除引用
                del model, clip, vae, sample
                # 强制 Python 垃圾回收 (处理循环引用)
                gc.collect()
                # 强制 PyTorch 释放缓存显存
                comfy.model_management.soft_empty_cache()

        # 最终合并
        if len(final_images_list) > 0:
            # cat 默认在 dim=0 合并: [1,H,W,C] + [1,H,W,C] -> [N,H,W,C]
            return (torch.cat(final_images_list, dim=0),)
        else:
            # 如果全部失败，返回一个空的黑色图片防止下游报错
            print("⚠️ [Multi-Ckpt] 所有模型均处理失败，返回空白图像。")
            empty_img = torch.zeros((1, height, width, 3), dtype=torch.float32)
            return (empty_img,)

NODE_CLASS_MAPPINGS = {
    "MultiCheckpointIncrementalNamer": MultiCheckpointIncrementalNamer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiCheckpointIncrementalNamer": "Multi-Checkpoint (Auto Incremental Name)"
}