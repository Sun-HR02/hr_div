#!/usr/bin/env python3
"""
DivPrune 前后对比可视化工具

用法:
python visualize_prune_comparison.py \
    --image_path /path/to/image.jpg \
    --model_path liuhaotian/llava-v1.5-7b \
    --prune_ratio 0.098 \
    --output_dir ./prune_comparison_results

功能:
1. 保存prune前后的视觉token特征
2. 可视化被保留和被剪枝的token位置
3. 生成注意力热力图对比
4. 统计分析报告
"""

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import json

# 添加LLaVA路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'LLaVA'))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


class PruneComparator:
    """DivPrune前后对比工具"""
    
    def __init__(self, model_path, device='cuda'):
        """初始化模型"""
        self.device = device
        self.model_name = get_model_name_from_path(model_path)
        
        print(f"Loading model: {model_path}")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=self.model_name,
            device_map=device
        )
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def load_and_process_image(self, image_path):
        """加载并处理图片"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model.config)
        
        if type(image_tensor) is list:
            image_tensor = [img.to(self.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.device, dtype=torch.float16)
        
        return image, image_tensor
    
    def extract_features_without_prune(self, image_tensor, prompt="Describe this image."):
        """提取未剪枝的特征"""
        # 临时禁用剪枝
        original_layer_index = os.environ.get('LAYER_INDEX', None)
        if 'LAYER_INDEX' in os.environ:
            del os.environ['LAYER_INDEX']
        
        try:
            # 准备输入
            conv = conv_templates["vicuna_v1"].copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()
            
            input_ids = tokenizer_image_token(
                prompt_text, 
                self.tokenizer, 
                IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0).to(self.device)
            
            # 前向传播获取特征
            with torch.no_grad():
                # 编码图像
                if type(image_tensor) is list:
                    image_features, attentions = self.model.encode_images(
                        torch.cat([img for img in image_tensor], dim=0),
                        return_attentions=True
                    )
                else:
                    image_features, attentions = self.model.encode_images(
                        image_tensor,
                        return_attentions=True
                    )
            
            # 展平特征
            if type(image_features) is list:
                image_features = image_features[0]
            
            if image_features.ndim == 3:
                image_features = image_features.squeeze(0)
            
            return {
                'features': image_features.cpu(),
                'attentions': attentions,
                'num_tokens': image_features.shape[0]
            }
        
        finally:
            # 恢复环境变量
            if original_layer_index is not None:
                os.environ['LAYER_INDEX'] = original_layer_index
    
    def extract_features_with_prune(self, image_tensor, prune_ratio, prompt="Describe this image."):
        """提取剪枝后的特征"""
        # 启用剪枝
        os.environ['LAYER_INDEX'] = '0'
        os.environ['SUBSET_RATIO'] = str(prune_ratio)
        
        try:
            # 准备输入
            conv = conv_templates["vicuna_v1"].copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()
            
            input_ids = tokenizer_image_token(
                prompt_text, 
                self.tokenizer, 
                IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0).to(self.device)
            
            # 前向传播获取特征和选中的索引
            with torch.no_grad():
                # 编码图像
                if type(image_tensor) is list:
                    image_features, attentions = self.model.encode_images(
                        torch.cat([img for img in image_tensor], dim=0),
                        return_attentions=True
                    )
                else:
                    image_features, attentions = self.model.encode_images(
                        image_tensor,
                        return_attentions=True
                    )
                
                # 展平特征
                if type(image_features) is list:
                    image_features = image_features[0]
                
                if image_features.ndim == 3:
                    image_features = image_features.squeeze(0)
                
                # 执行DivPrune获取选中的索引
                img_feature_len = image_features.shape[0]
                selected_indices, cosine_matrix = self.model.DivPrune(
                    image_features,
                    img_feature_len,
                    threshold_ratio=prune_ratio,
                    attentions=attentions
                )
            
            return {
                'features': image_features.cpu(),
                'selected_indices': selected_indices.cpu(),
                'attentions': attentions,
                'num_tokens': image_features.shape[0],
                'num_selected': len(selected_indices),
                'cosine_matrix': cosine_matrix.cpu() if cosine_matrix is not None else None
            }
        
        finally:
            # 清理环境变量
            if 'LAYER_INDEX' in os.environ:
                del os.environ['LAYER_INDEX']
            if 'SUBSET_RATIO' in os.environ:
                del os.environ['SUBSET_RATIO']
    
    def visualize_token_selection(self, original_data, pruned_data, output_path):
        """可视化token选择结果"""
        num_tokens = original_data['num_tokens']
        selected_indices = pruned_data['selected_indices'].numpy()
        
        # 计算grid大小 (假设是方形)
        grid_size = int(np.sqrt(num_tokens))
        
        # 创建mask
        mask = np.zeros(num_tokens)
        mask[selected_indices] = 1
        mask = mask.reshape(grid_size, grid_size)
        
        # 绘图
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图：选中的token
        axes[0].imshow(mask, cmap='RdYlGn', vmin=0, vmax=1)
        axes[0].set_title(f'Selected Tokens (Green)\n{pruned_data["num_selected"]}/{num_tokens} tokens kept')
        axes[0].axis('off')
        
        # 右图：token索引
        token_indices = np.arange(num_tokens).reshape(grid_size, grid_size)
        im = axes[1].imshow(token_indices, cmap='viridis')
        axes[1].set_title('Token Index Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Token selection visualization saved to: {output_path}")
    
    def visualize_attention_comparison(self, original_data, pruned_data, output_path):
        """可视化注意力对比"""
        if original_data['attentions'] is None:
            print("No attention data available")
            return
        
        # 提取CLS token的注意力
        orig_attn = original_data['attentions'][-1]  # 最后一层
        orig_cls_attn = orig_attn[:, :, 0, 1:].mean(dim=1).squeeze(0).cpu().numpy()
        
        pruned_attn = pruned_data['attentions'][-1]
        pruned_cls_attn = pruned_attn[:, :, 0, 1:].mean(dim=1).squeeze(0).cpu().numpy()
        
        # 计算grid大小
        grid_size = int(np.sqrt(len(orig_cls_attn)))
        orig_cls_attn = orig_cls_attn.reshape(grid_size, grid_size)
        pruned_cls_attn = pruned_cls_attn.reshape(grid_size, grid_size)
        
        # 绘图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 原始注意力
        im0 = axes[0].imshow(orig_cls_attn, cmap='hot')
        axes[0].set_title('Original Attention (CLS to Patches)')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0])
        
        # 剪枝后注意力
        im1 = axes[1].imshow(pruned_cls_attn, cmap='hot')
        axes[1].set_title('Pruned Attention (CLS to Patches)')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # 差异
        diff = np.abs(orig_cls_attn - pruned_cls_attn)
        im2 = axes[2].imshow(diff, cmap='coolwarm')
        axes[2].set_title('Attention Difference (Absolute)')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Attention comparison saved to: {output_path}")
    
    def visualize_feature_similarity(self, original_data, pruned_data, output_path):
        """可视化特征相似度矩阵"""
        if pruned_data['cosine_matrix'] is None:
            print("No cosine matrix available")
            return
        
        cosine_matrix = pruned_data['cosine_matrix'].numpy()
        
        # 绘制相似度矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cosine_matrix, cmap='coolwarm', center=0.5, 
                   xticklabels=False, yticklabels=False)
        plt.title('Token Diversity Matrix (1 - Cosine Similarity)')
        plt.xlabel('Token Index')
        plt.ylabel('Token Index')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature similarity matrix saved to: {output_path}")
    
    def generate_statistics_report(self, original_data, pruned_data, output_path):
        """生成统计报告"""
        selected_indices = pruned_data['selected_indices'].numpy()
        
        # 计算统计信息
        stats = {
            'original_num_tokens': original_data['num_tokens'],
            'pruned_num_tokens': pruned_data['num_selected'],
            'prune_ratio': 1 - (pruned_data['num_selected'] / original_data['num_tokens']),
            'selected_indices': selected_indices.tolist(),
            'feature_dim': original_data['features'].shape[1],
        }
        
        # 如果有注意力数据，计算注意力统计
        if original_data['attentions'] is not None:
            orig_attn = original_data['attentions'][-1][:, :, 0, 1:].mean(dim=1).squeeze(0).cpu().numpy()
            selected_attn = orig_attn[selected_indices]
            
            stats['attention_stats'] = {
                'original_mean': float(orig_attn.mean()),
                'original_std': float(orig_attn.std()),
                'selected_mean': float(selected_attn.mean()),
                'selected_std': float(selected_attn.std()),
                'selected_min': float(selected_attn.min()),
                'selected_max': float(selected_attn.max()),
            }
        
        # 保存为JSON
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 打印报告
        print("\n" + "="*60)
        print("PRUNING STATISTICS REPORT")
        print("="*60)
        print(f"Original tokens: {stats['original_num_tokens']}")
        print(f"Pruned tokens: {stats['pruned_num_tokens']}")
        print(f"Prune ratio: {stats['prune_ratio']:.2%}")
        print(f"Feature dimension: {stats['feature_dim']}")
        
        if 'attention_stats' in stats:
            print("\nAttention Statistics:")
            print(f"  Original - Mean: {stats['attention_stats']['original_mean']:.4f}, "
                  f"Std: {stats['attention_stats']['original_std']:.4f}")
            print(f"  Selected - Mean: {stats['attention_stats']['selected_mean']:.4f}, "
                  f"Std: {stats['attention_stats']['selected_std']:.4f}")
            print(f"  Selected - Min: {stats['attention_stats']['selected_min']:.4f}, "
                  f"Max: {stats['attention_stats']['selected_max']:.4f}")
        
        print("="*60)
        print(f"\nFull report saved to: {output_path}")
    
    def compare_image(self, image_path, prune_ratio, output_dir, prompt="Describe this image."):
        """对比指定图片的prune前后差异"""
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存原始图片
        original_image, image_tensor = self.load_and_process_image(image_path)
        original_image.save(output_dir / "original_image.jpg")
        
        print(f"\nProcessing image: {image_path}")
        print(f"Prune ratio: {prune_ratio}")
        
        # 提取未剪枝特征
        print("\n[1/2] Extracting features WITHOUT pruning...")
        original_data = self.extract_features_without_prune(image_tensor, prompt)
        
        # 提取剪枝后特征
        print("[2/2] Extracting features WITH pruning...")
        pruned_data = self.extract_features_with_prune(image_tensor, prune_ratio, prompt)
        
        # 生成可视化
        print("\nGenerating visualizations...")
        self.visualize_token_selection(
            original_data, pruned_data,
            output_dir / "token_selection.png"
        )
        
        self.visualize_attention_comparison(
            original_data, pruned_data,
            output_dir / "attention_comparison.png"
        )
        
        self.visualize_feature_similarity(
            original_data, pruned_data,
            output_dir / "feature_similarity.png"
        )
        
        # 生成统计报告
        self.generate_statistics_report(
            original_data, pruned_data,
            output_dir / "statistics.json"
        )
        
        print(f"\n✓ All results saved to: {output_dir}")
        print("\nGenerated files:")
        print(f"  - original_image.jpg: 原始图片")
        print(f"  - token_selection.png: token选择可视化")
        print(f"  - attention_comparison.png: 注意力对比")
        print(f"  - feature_similarity.png: 特征相似度矩阵")
        print(f"  - statistics.json: 统计报告")


def main():
    parser = argparse.ArgumentParser(description='DivPrune前后对比可视化工具')
    parser.add_argument('--image_path', type=str, required=True,
                       help='输入图片路径')
    parser.add_argument('--model_path', type=str, 
                       default='liuhaotian/llava-v1.5-7b',
                       help='模型路径或HuggingFace模型ID')
    parser.add_argument('--prune_ratio', type=float, default=0.098,
                       help='剪枝比例 (default: 0.098)')
    parser.add_argument('--output_dir', type=str, 
                       default='./prune_comparison_results',
                       help='输出目录')
    parser.add_argument('--prompt', type=str, 
                       default='Describe this image.',
                       help='输入提示词')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 检查图片是否存在
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        return
    
    # 创建对比器
    comparator = PruneComparator(args.model_path, args.device)
    
    # 执行对比
    comparator.compare_image(
        args.image_path,
        args.prune_ratio,
        args.output_dir,
        args.prompt
    )


if __name__ == '__main__':
    main()
