#!/usr/bin/env python3
"""
将 HuggingFace BERT 权重转换为 BERT-pytorch 格式（完全正确版本）
基于 BERT-pytorch 实际源代码的准确映射
"""

import torch
import sys
import os
import argparse

def detect_model_config(hf_state_dict):
    """
    从 HuggingFace state_dict 自动检测模型配置

    Returns:
        dict: 包含 vocab_size, hidden, n_layers, attn_heads
    """
    # 检测 vocab_size
    vocab_size = hf_state_dict['bert.embeddings.word_embeddings.weight'].shape[0]

    # 检测 hidden size
    hidden = hf_state_dict['bert.embeddings.word_embeddings.weight'].shape[1]

    # 检测层数
    n_layers = 0
    for key in hf_state_dict.keys():
        if 'bert.encoder.layer.' in key:
            layer_num = int(key.split('.')[3])
            n_layers = max(n_layers, layer_num + 1)

    # 检测注意力头数 (通过 query 权重形状推断)
    # query weight shape: [hidden, hidden]
    # attn_heads = hidden / head_dim, head_dim = hidden / attn_heads
    # 对于 BERT，通常 head_dim = 64
    attn_heads = hidden // 64

    model_type = "BERT-Large" if n_layers == 24 else "BERT-Base"

    config = {
        'vocab_size': vocab_size,
        'hidden': hidden,
        'n_layers': n_layers,
        'attn_heads': attn_heads,
        'model_type': model_type
    }

    return config

def convert_huggingface_to_bert_pytorch(hf_state_dict, model_config=None):
    """
    将 HuggingFace 格式的权重转换为 BERT-pytorch 格式

    关键差异（全部已修复）：
    1. Position Embedding: 已修复为学习的 nn.Embedding ✓
    2. Embedding LayerNorm: 已添加 ✓
    3. LayerNorm 参数: a_2/b_2 而不是 weight/bias
    4. Attention 路径: attention.linear_layers 而不是 attention.attention.linear_layers
    5. Sublayer: input_sublayer 和 output_sublayer

    Args:
        hf_state_dict: HuggingFace 模型的 state_dict
        model_config: 模型配置字典 (可选，将自动检测)
    """

    # 自动检测模型配置
    if model_config is None:
        model_config = detect_model_config(hf_state_dict)
        print(f"\n📊 自动检测模型配置:")
        print(f"  模型类型:     {model_config['model_type']}")
        print(f"  词汇表大小:   {model_config['vocab_size']}")
        print(f"  隐藏层维度:   {model_config['hidden']}")
        print(f"  Transformer层: {model_config['n_layers']}")
        print(f"  注意力头数:   {model_config['attn_heads']}")

    converted = {}
    converted_count = 0
    skipped_count = 0

    print("\n开始转换权重...")
    print("=" * 80)

    for key, value in hf_state_dict.items():
        new_key = None

        # ===== Embeddings =====

        # Token Embedding
        if key == 'bert.embeddings.word_embeddings.weight':
            new_key = 'embedding.token.weight'

        # Position Embedding - 现在转换（已修复为学习的 Embedding）
        elif key == 'bert.embeddings.position_embeddings.weight':
            new_key = 'embedding.position.pe.weight'
            print(f"✓ {key:70s} -> {new_key}")
            print(f"   📝 位置编码现在是可学习的 nn.Embedding（已修复）")

        # Segment Embedding
        elif key == 'bert.embeddings.token_type_embeddings.weight':
            new_key = 'embedding.segment.weight'
            # HuggingFace: [2, 768] -> BERT-pytorch: [3, 768]
            if value.shape[0] == 2:
                print(f"📝 扩展 segment embedding: {value.shape} -> [3, {value.shape[1]}]")
                new_value = torch.zeros(3, value.shape[1])
                new_value[1] = value[0]  # 句子 A
                new_value[2] = value[1]  # 句子 B
                value = new_value

        # Embedding LayerNorm - 现在转换（已添加到 BERTEmbedding）
        elif key in ['bert.embeddings.LayerNorm.weight', 'bert.embeddings.LayerNorm.gamma']:
            new_key = 'embedding.layer_norm.weight'
        elif key in ['bert.embeddings.LayerNorm.bias', 'bert.embeddings.LayerNorm.beta']:
            new_key = 'embedding.layer_norm.bias'

        # ===== Encoder Layers =====

        elif 'bert.encoder.layer.' in key:
            parts = key.split('.')
            layer_num = parts[3]  # bert.encoder.layer.{i}...

            # --- Attention: Query, Key, Value ---
            if 'attention.self.query.weight' in key:
                new_key = f'transformer_blocks.{layer_num}.attention.linear_layers.0.weight'
            elif 'attention.self.query.bias' in key:
                new_key = f'transformer_blocks.{layer_num}.attention.linear_layers.0.bias'

            elif 'attention.self.key.weight' in key:
                new_key = f'transformer_blocks.{layer_num}.attention.linear_layers.1.weight'
            elif 'attention.self.key.bias' in key:
                new_key = f'transformer_blocks.{layer_num}.attention.linear_layers.1.bias'

            elif 'attention.self.value.weight' in key:
                new_key = f'transformer_blocks.{layer_num}.attention.linear_layers.2.weight'
            elif 'attention.self.value.bias' in key:
                new_key = f'transformer_blocks.{layer_num}.attention.linear_layers.2.bias'

            # --- Attention Output Dense ---
            elif 'attention.output.dense.weight' in key:
                new_key = f'transformer_blocks.{layer_num}.attention.output_linear.weight'
            elif 'attention.output.dense.bias' in key:
                new_key = f'transformer_blocks.{layer_num}.attention.output_linear.bias'

            # --- Attention Output LayerNorm (input_sublayer) ---
            elif 'attention.output.LayerNorm.weight' in key or 'attention.output.LayerNorm.gamma' in key:
                new_key = f'transformer_blocks.{layer_num}.input_sublayer.norm.a_2'
            elif 'attention.output.LayerNorm.bias' in key or 'attention.output.LayerNorm.beta' in key:
                new_key = f'transformer_blocks.{layer_num}.input_sublayer.norm.b_2'

            # --- Feed Forward: Intermediate ---
            elif 'intermediate.dense.weight' in key:
                new_key = f'transformer_blocks.{layer_num}.feed_forward.w_1.weight'
            elif 'intermediate.dense.bias' in key:
                new_key = f'transformer_blocks.{layer_num}.feed_forward.w_1.bias'

            # --- Feed Forward: Output Dense ---
            elif 'output.dense.weight' in key:
                new_key = f'transformer_blocks.{layer_num}.feed_forward.w_2.weight'
            elif 'output.dense.bias' in key:
                new_key = f'transformer_blocks.{layer_num}.feed_forward.w_2.bias'

            # --- Feed Forward Output LayerNorm (output_sublayer) ---
            elif 'output.LayerNorm.weight' in key or 'output.LayerNorm.gamma' in key:
                new_key = f'transformer_blocks.{layer_num}.output_sublayer.norm.a_2'
            elif 'output.LayerNorm.bias' in key or 'output.LayerNorm.beta' in key:
                new_key = f'transformer_blocks.{layer_num}.output_sublayer.norm.b_2'

        # ===== Pooler & Classification Heads (跳过) =====

        elif 'bert.pooler' in key:
            skipped_count += 1
            continue

        elif key.startswith('cls.'):
            skipped_count += 1
            continue

        else:
            print(f"⚠️  未知的 key: {key}")
            skipped_count += 1
            continue

        # 保存转换后的权重
        if new_key:
            converted[new_key] = value
            converted_count += 1

            # 只打印前 10 个和最后 10 个
            if converted_count <= 10 or converted_count > 180:
                print(f"✓ {key:70s} -> {new_key}")
            elif converted_count == 11:
                print(f"... (省略中间的转换信息)")

    print("=" * 80)
    print(f"转换完成:")
    print(f"  ✅ 成功转换: {converted_count} 个参数")
    print(f"  ⏭️  跳过: {skipped_count} 个参数")
    print(f"  📊 总计: {len(hf_state_dict)} 个参数")

    # 根据模型类型计算期望参数数量
    # BERT-Base: 195, BERT-Large: 387
    expected_params = 195 if model_config['n_layers'] == 12 else 387
    print(f"\n期望模型能加载: {converted_count}/{expected_params} 个参数")

    return converted, model_config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='HuggingFace BERT 权重转换工具 (支持 BERT-Base 和 BERT-Large)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动检测模型配置
  python3 convert_hf_to_bert_pytorch.py bert-base-uncased.bin
  python3 convert_hf_to_bert_pytorch.py bert-large-uncased.bin

  # 手动指定配置
  python3 convert_hf_to_bert_pytorch.py bert-large.bin bert-large-converted.bin --hidden 1024 --layers 24 --heads 16
        """
    )

    parser.add_argument('input', help='输入权重文件 (.bin)')
    parser.add_argument('output', nargs='?', help='输出权重文件 (.bin，可选)')
    parser.add_argument('--vocab_size', type=int, help='词汇表大小 (默认: 自动检测)')
    parser.add_argument('--hidden', type=int, help='隐藏层维度 (默认: 自动检测)')
    parser.add_argument('--layers', type=int, help='Transformer层数 (默认: 自动检测)')
    parser.add_argument('--heads', type=int, help='注意力头数 (默认: 自动检测)')

    args = parser.parse_args()

    # 设置默认输出路径
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}-converted.bin"

    return args

def main():
    """主函数"""
    print("=" * 80)
    print("HuggingFace BERT 权重转换工具 (支持 BERT-Base 和 BERT-Large)")
    print("基于 BERT-pytorch 实际源代码结构")
    print("=" * 80)

    # 解析命令行参数
    args = parse_args()
    input_path = args.input
    output_path = args.output

    print(f"\n输入文件: {input_path}")
    print(f"输出文件: {output_path}")

    # 检查输入文件
    if not os.path.exists(input_path):
        print(f"\n❌ 错误: 输入文件不存在: {input_path}")
        return 1

    # 加载 HuggingFace 权重
    print(f"\n📥 加载 HuggingFace 权重...")
    try:
        hf_state_dict = torch.load(input_path, map_location='cpu', weights_only=False)
        print(f"✅ 成功加载 {len(hf_state_dict)} 个参数")

        # 检测命名格式
        has_gamma_beta = any('gamma' in k or 'beta' in k for k in hf_state_dict.keys())
        if has_gamma_beta:
            print(f"   📌 检测到旧版 LayerNorm 命名 (gamma/beta)")
        else:
            print(f"   📌 检测到新版 LayerNorm 命名 (weight/bias)")

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return 1

    # 转换权重
    try:
        # 如果用户指定了配置，使用用户配置；否则自动检测
        if any([args.vocab_size, args.hidden, args.layers, args.heads]):
            # 先自动检测，然后用用户参数覆盖
            model_config = detect_model_config(hf_state_dict)
            if args.vocab_size:
                model_config['vocab_size'] = args.vocab_size
            if args.hidden:
                model_config['hidden'] = args.hidden
            if args.layers:
                model_config['n_layers'] = args.layers
            if args.heads:
                model_config['attn_heads'] = args.heads

            print(f"\n📊 使用指定的模型配置:")
            print(f"  词汇表大小:   {model_config['vocab_size']}")
            print(f"  隐藏层维度:   {model_config['hidden']}")
            print(f"  Transformer层: {model_config['n_layers']}")
            print(f"  注意力头数:   {model_config['attn_heads']}")

            converted_state_dict, model_config = convert_huggingface_to_bert_pytorch(hf_state_dict, model_config)
        else:
            # 完全自动检测
            converted_state_dict, model_config = convert_huggingface_to_bert_pytorch(hf_state_dict)
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 保存转换后的权重
    print(f"\n💾 保存转换后的权重到: {output_path}")
    try:
        torch.save(converted_state_dict, output_path)
        print(f"✅ 保存成功")
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return 1

    # 验证
    print(f"\n" + "=" * 80)
    print("验证转换结果")
    print("=" * 80)

    try:
        # 创建 BERT 模型进行验证
        sys.path.insert(0, '.')
        from bert_pytorch import BERT

        print(f"\n使用配置创建验证模型:")
        print(f"  vocab_size={model_config['vocab_size']}, hidden={model_config['hidden']}, "
              f"n_layers={model_config['n_layers']}, attn_heads={model_config['attn_heads']}")

        model = BERT(
            vocab_size=model_config['vocab_size'],
            hidden=model_config['hidden'],
            n_layers=model_config['n_layers'],
            attn_heads=model_config['attn_heads'],
            dropout=0.1
        )
        model_dict = model.state_dict()

        loaded = torch.load(output_path, map_location='cpu', weights_only=False)

        print(f"✅ 转换后的权重: {len(loaded)} 个参数")
        print(f"✅ 模型期望: {len(model_dict)} 个参数")

        # 检查匹配度
        matched = set(loaded.keys()) & set(model_dict.keys())
        print(f"✅ 匹配的参数: {len(matched)} 个")

        if len(matched) != len(loaded):
            print(f"⚠️  警告: {len(loaded) - len(matched)} 个参数未匹配")

        # 尝试加载
        result = model.load_state_dict(loaded, strict=False)

        if result.missing_keys:
            print(f"\n⚠️  模型中缺失的参数: {len(result.missing_keys)} 个")
            for key in result.missing_keys[:5]:
                print(f"     - {key}")
            if len(result.missing_keys) > 5:
                print(f"     ... 还有 {len(result.missing_keys) - 5} 个")

        if result.unexpected_keys:
            print(f"\n⚠️  转换文件中多余的参数: {len(result.unexpected_keys)} 个")
            for key in result.unexpected_keys[:5]:
                print(f"     - {key}")

        if not result.missing_keys and not result.unexpected_keys:
            print(f"\n🎉 完美！所有参数都正确匹配！")

    except Exception as e:
        print(f"⚠️  验证跳过（无法导入 BERT 模型）: {e}")

    print(f"\n" + "=" * 80)
    print("✅ 转换完成！")
    print("=" * 80)
    print(f"\n现在可以运行推理脚本:")
    print(f"  python3 bert_real_inference.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
