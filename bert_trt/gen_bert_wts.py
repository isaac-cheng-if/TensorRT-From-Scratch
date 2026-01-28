#!/usr/bin/env python3
"""
BERT Weight Converter
Converts PyTorch BERT model weights to .wts format for TensorRT
"""

import sys
import argparse
import os
import struct
import torch
import torch.nn as nn

# Import BERT model from bert_standalone.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bert_standalone import BERT

def detect_model_config_from_weights(weights_file):
    """
    从权重文件中检测模型配置

    Returns:
        dict: 包含 vocab_size, hidden, n_layers, attn_heads
    """
    print(f"🔍 检测权重文件的模型配置...")

    checkpoint = torch.load(weights_file, map_location='cpu', weights_only=False)

    # 检测 vocab_size
    if 'embedding.token.weight' in checkpoint:
        vocab_size = checkpoint['embedding.token.weight'].shape[0]
        hidden = checkpoint['embedding.token.weight'].shape[1]
    elif 'bert.embeddings.word_embeddings.weight' in checkpoint:
        vocab_size = checkpoint['bert.embeddings.word_embeddings.weight'].shape[0]
        hidden = checkpoint['bert.embeddings.word_embeddings.weight'].shape[1]
    else:
        raise ValueError("无法检测模型配置: 找不到 embedding 层")

    # 检测层数
    n_layers = 0
    for key in checkpoint.keys():
        if 'transformer_blocks.' in key:
            layer_num = int(key.split('.')[1])
            n_layers = max(n_layers, layer_num + 1)
        elif 'bert.encoder.layer.' in key:
            layer_num = int(key.split('.')[3])
            n_layers = max(n_layers, layer_num + 1)

    # 检测注意力头数 (通常 head_dim = 64)
    attn_heads = hidden // 64

    model_type = "BERT-Large" if n_layers == 24 else "BERT-Base"

    config = {
        'vocab_size': vocab_size,
        'hidden': hidden,
        'n_layers': n_layers,
        'attn_heads': attn_heads,
        'model_type': model_type
    }

    print(f"  检测到: {model_type}")
    print(f"  - 词汇表: {vocab_size}")
    print(f"  - 隐藏层: {hidden}")
    print(f"  - 层数: {n_layers}")
    print(f"  - 注意力头: {attn_heads}")

    return config

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert BERT .bin file to .wts (支持 BERT-Base 和 BERT-Large)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动检测模型配置
  python3 gen_bert_wts.py -w bert-base-uncased-converted.bin
  python3 gen_bert_wts.py -w bert-large-uncased-converted.bin -o bert-large.wts

  # 使用预设配置
  python3 gen_bert_wts.py -w weights.bin --model-type bert-large

  # 手动指定配置
  python3 gen_bert_wts.py -w weights.bin --hidden 1024 --layers 24 --heads 16
        """
    )
    parser.add_argument('-w', '--weights', required=True,
                        help='Input weights (.bin) file path (required)')
    parser.add_argument('-o', '--output', help='Output (.wts) file path (optional)')
    parser.add_argument('--model-type', choices=['bert-base', 'bert-large', 'auto'],
                        default='auto',
                        help='Model type preset (default: auto-detect)')
    parser.add_argument('-v', '--vocab_size', type=int,
                        help='Vocabulary size (default: auto-detect)')
    parser.add_argument('--hidden', type=int,
                        help='Hidden dimension (default: auto-detect)')
    parser.add_argument('--layers', type=int,
                        help='Number of transformer layers (default: auto-detect)')
    parser.add_argument('--heads', type=int,
                        help='Number of attention heads (default: auto-detect)')
    args = parser.parse_args()
    
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid input file')
    
    if not args.output:
        args.output = os.path.splitext(args.weights)[0] + '.wts'
    elif os.path.isdir(args.output):
        args.output = os.path.join(
            args.output,
            os.path.splitext(os.path.basename(args.weights))[0] + '.wts')
    
    return args

def main():
    args = parse_args()

    print(f'=' * 70)
    print(f'BERT Weight Converter (.bin -> .wts)')
    print(f'=' * 70)
    print(f'Loading {args.weights}')

    # 确定模型配置
    if args.model_type == 'bert-base':
        config = {
            'vocab_size': 30522,
            'hidden': 768,
            'n_layers': 12,
            'attn_heads': 12,
            'model_type': 'BERT-Base'
        }
        print(f'\n使用 BERT-Base 预设配置')
    elif args.model_type == 'bert-large':
        config = {
            'vocab_size': 30522,
            'hidden': 1024,
            'n_layers': 24,
            'attn_heads': 16,
            'model_type': 'BERT-Large'
        }
        print(f'\n使用 BERT-Large 预设配置')
    else:
        # 自动检测
        config = detect_model_config_from_weights(args.weights)

    # 用户指定的参数覆盖
    if args.vocab_size:
        config['vocab_size'] = args.vocab_size
    if args.hidden:
        config['hidden'] = args.hidden
    if args.layers:
        config['n_layers'] = args.layers
    if args.heads:
        config['attn_heads'] = args.heads

    print(f'\n📊 最终模型配置:')
    print(f"  模型类型: {config.get('model_type', 'Custom')}")
    print(f"  词汇表:   {config['vocab_size']}")
    print(f"  隐藏层:   {config['hidden']}")
    print(f"  层数:     {config['n_layers']}")
    print(f"  注意力头: {config['attn_heads']}")

    # Create BERT model
    model = BERT(
        vocab_size=config['vocab_size'],
        hidden=config['hidden'],
        n_layers=config['n_layers'],
        attn_heads=config['attn_heads'],
        dropout=0.1
    )
    
    # Load weights
    device = 'cpu'
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    
    # Check if it's already in our format or HuggingFace format
    sample_key = list(checkpoint.keys())[0]
    if sample_key.startswith('embedding.') or sample_key.startswith('transformer_blocks.'):
        print('Detected converted format, loading directly...')
        model.load_state_dict(checkpoint, strict=False)
    else:
        print('Detected HuggingFace format, converting...')
        from bert_standalone import load_huggingface_weights
        # Save current state dict
        temp_file = '/tmp/bert_temp.bin'
        torch.save(checkpoint, temp_file)
        load_huggingface_weights(model, temp_file)
        os.remove(temp_file)
    
    model.to(device).eval()
    
    print(f'Model loaded successfully')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Convert to .wts format
    print(f'Converting to .wts format...')
    state_dict = model.state_dict()
    
    with open(args.output, 'w') as f:
        f.write('{}\n'.format(len(state_dict.keys())))
        for k, v in state_dict.items():
            # 保持原始张量形状，按行优先顺序展平
            vr = v.flatten().cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')
    
    print(f'✅ .wts file generated: {args.output}')
    print(f'Total tensors: {len(state_dict.keys())}')
    
    # Print some key tensor shapes for verification
    print('\nKey tensor shapes:')
    print(f'  Token embedding: {state_dict["embedding.token.weight"].shape}')
    print(f'  Position embedding: {state_dict["embedding.position.pe.weight"].shape}')
    print(f'  Segment embedding: {state_dict["embedding.segment.weight"].shape}')
    print(f'  First layer attention QKV: {state_dict["transformer_blocks.0.attention.linear_layers.0.weight"].shape}')

if __name__ == '__main__':
    main()

