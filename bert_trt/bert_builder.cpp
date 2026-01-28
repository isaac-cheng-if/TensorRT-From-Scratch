/*
 * BERT Network Builder
 * Handles network construction, weight loading, and engine serialization
 */

#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>
#include <cassert>
#include <string>

#include "NvInfer.h"
#include "cuda_runtime_api.h"

// ==================== Configuration ====================
namespace Config {
    // 静态配置（不随模型变化）
    const int BATCH_SIZE = 16;
    const int MAX_SEQ_LEN = 128;
    const int VOCAB_SIZE = 30522;
    const char* INPUT_IDS_NAME = "input_ids";
    const char* ATTENTION_MASK_NAME = "attention_mask";
    const char* SEGMENT_IDS_NAME = "token_type_ids";  // 改为与 ONNX 一致
    const char* OUTPUT_NAME = "output";
    const int GPU_ID = 0;
}

// ==================== CUDA Error Checking ====================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// ==================== TensorRT Logger ====================
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

// ==================== 常量管理器 ====================
class ConstantManager {
private:
    std::map<float, std::unique_ptr<float[]>> constants;
    
public:
    nvinfer1::Weights getConstant(float value) {
        if (constants.find(value) == constants.end()) {
            constants[value] = std::unique_ptr<float[]>(new float[1]{value});
        }
        return nvinfer1::Weights{nvinfer1::DataType::kFLOAT, constants[value].get(), 1};
    }
    
    ~ConstantManager() = default;
};

ConstantManager g_ConstantManager;

// ==================== Weight Loading ====================
std::map<std::string, nvinfer1::Weights> LoadWeights(const std::string& file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, nvinfer1::Weights> weightMap;

    std::ifstream input(file);
    if (!input.is_open()) {
        std::cerr << "ERROR: Unable to load weight file: " << file << std::endl;
        exit(1);
    }

    int32_t count;
    input >> count;
    if (count <= 0) {
        std::cerr << "ERROR: Invalid weight count" << std::endl;
        exit(1);
    }

    while (count--) {
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        uint32_t size;
        std::string name;
        input >> name >> std::dec >> size;

        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(uint32_t) * size));
        for (uint32_t x = 0; x < size; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        wt.count = size;
        weightMap[name] = wt;
    }

    std::cout << "Loaded " << weightMap.size() << " weight tensors" << std::endl;
    return weightMap;
}

// ==================== BERT Network Building Blocks ====================

// Helper: Fully Connected Layer using MatrixMultiply
nvinfer1::ITensor* FullyConnected3D(nvinfer1::INetworkDefinition* network,
                                    nvinfer1::ITensor& input,
                                    int output_size,
                                    nvinfer1::Weights weight,
                                    nvinfer1::Weights bias) {
    auto input_dims = input.getDimensions();
    int input_size = input_dims.d[input_dims.nbDims - 1];
    
    // For 3D input [B, N, D_in]
    if (input_dims.nbDims == 3) {
        int batch = input_dims.d[0];
        int seq_len = input_dims.d[1];
        
        // Reshape input to [B*N, D_in]
        auto input_reshape = network->addShuffle(input);
        nvinfer1::Dims flatten_dims;
        flatten_dims.nbDims = 2;
        flatten_dims.d[0] = batch * seq_len;
        flatten_dims.d[1] = input_size;
        input_reshape->setReshapeDimensions(flatten_dims);
        
        // Weight as constant: [D_out, D_in]
        nvinfer1::Dims weight_dims;
        weight_dims.nbDims = 2;
        weight_dims.d[0] = output_size;
        weight_dims.d[1] = input_size;
        auto weight_const = network->addConstant(weight_dims, weight);
        
        // MatMul: [B*N, D_in] @ [D_out, D_in]^T = [B*N, D_out]
        auto matmul = network->addMatrixMultiply(*input_reshape->getOutput(0),
                                                 nvinfer1::MatrixOperation::kNONE,
                                                 *weight_const->getOutput(0),
                                                 nvinfer1::MatrixOperation::kTRANSPOSE);
        
        // Add bias: [B*N, D_out] + [D_out]
        nvinfer1::Dims bias_dims;
        bias_dims.nbDims = 2;
        bias_dims.d[0] = 1;
        bias_dims.d[1] = output_size;
        auto bias_const = network->addConstant(bias_dims, bias);
        auto add_bias = network->addElementWise(*matmul->getOutput(0),
                                               *bias_const->getOutput(0),
                                               nvinfer1::ElementWiseOperation::kSUM);
        
        // Reshape back to [B, N, D_out]
        auto output_reshape = network->addShuffle(*add_bias->getOutput(0));
        nvinfer1::Dims output_dims;
        output_dims.nbDims = 3;
        output_dims.d[0] = batch;
        output_dims.d[1] = seq_len;
        output_dims.d[2] = output_size;
        output_reshape->setReshapeDimensions(output_dims);
        
        return output_reshape->getOutput(0);
    } else {
        // For 2D input use standard FC
        auto fc = network->addFullyConnected(input, output_size, weight, bias);
        assert(fc);
        return fc->getOutput(0);
    }
}

// GELU Activation
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
nvinfer1::ITensor* GELU(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    // x³
    auto x_pow2 = network->addElementWise(input, input, nvinfer1::ElementWiseOperation::kPROD);
    auto x_pow3 = network->addElementWise(*x_pow2->getOutput(0), input, nvinfer1::ElementWiseOperation::kPROD);
    
    // 0.044715 * x³
    auto coeff_const = network->addConstant(nvinfer1::Dims3{1, 1, 1}, g_ConstantManager.getConstant(coeff));
    auto coeff_mul = network->addElementWise(*x_pow3->getOutput(0), *coeff_const->getOutput(0), 
                                             nvinfer1::ElementWiseOperation::kPROD);
    
    // x + 0.044715 * x³
    auto sum1 = network->addElementWise(input, *coeff_mul->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    
    // sqrt(2/π) * (x + 0.044715 * x³)
    auto sqrt_const = network->addConstant(nvinfer1::Dims3{1, 1, 1}, g_ConstantManager.getConstant(sqrt_2_over_pi));
    auto sqrt_mul = network->addElementWise(*sum1->getOutput(0), *sqrt_const->getOutput(0),
                                            nvinfer1::ElementWiseOperation::kPROD);
    
    // tanh(...)
    auto tanh_layer = network->addActivation(*sqrt_mul->getOutput(0), nvinfer1::ActivationType::kTANH);
    
    // 1 + tanh(...)
    auto one_const = network->addConstant(nvinfer1::Dims3{1, 1, 1}, g_ConstantManager.getConstant(1.0f));
    auto add_one = network->addElementWise(*tanh_layer->getOutput(0), *one_const->getOutput(0),
                                           nvinfer1::ElementWiseOperation::kSUM);
    
    // x * (1 + tanh(...))
    auto mul1 = network->addElementWise(input, *add_one->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    
    // 0.5 * x * (1 + tanh(...))
    auto half_const = network->addConstant(nvinfer1::Dims3{1, 1, 1}, g_ConstantManager.getConstant(0.5f));
    auto final_mul = network->addElementWise(*mul1->getOutput(0), *half_const->getOutput(0),
                                             nvinfer1::ElementWiseOperation::kPROD);
    
    return final_mul->getOutput(0);
}

// Layer Normalization (custom implementation using LayerNorm class parameters)
nvinfer1::ILayer* LayerNorm(nvinfer1::INetworkDefinition* network,
                            std::map<std::string, nvinfer1::Weights>& weightMap,
                            nvinfer1::ITensor& input,
                            int dim,
                            const std::string& lname,
                            float eps = 1e-12) {
    
    float* gamma = (float*)weightMap[lname + ".a_2"].values;
    float* beta = (float*)weightMap[lname + ".b_2"].values;
    
    // 1. 计算均值 (沿着最后一个维度)
    uint32_t reduceAxes = 1 << 2; // 第2维 (batch=0, seq=1, dim=2)
    auto mean_layer = network->addReduce(input, nvinfer1::ReduceOperation::kAVG, reduceAxes, true);
    assert(mean_layer);
    
    // 2. x - mean
    auto x_sub_mean = network->addElementWise(input, *mean_layer->getOutput(0), 
                                              nvinfer1::ElementWiseOperation::kSUB);
    
    // 3. (x - mean)^2
    auto x_sub_mean_sq = network->addElementWise(*x_sub_mean->getOutput(0), 
                                                  *x_sub_mean->getOutput(0), 
                                                  nvinfer1::ElementWiseOperation::kPROD);
    
    // 4. 计算方差 variance = mean((x - mean)^2)
    auto variance = network->addReduce(*x_sub_mean_sq->getOutput(0), 
                                      nvinfer1::ReduceOperation::kAVG, reduceAxes, true);
    
    // 5. variance + eps
    auto eps_layer = network->addConstant(nvinfer1::Dims3{1, 1, 1}, g_ConstantManager.getConstant(eps));
    auto var_plus_eps = network->addElementWise(*variance->getOutput(0), 
                                                *eps_layer->getOutput(0), 
                                                nvinfer1::ElementWiseOperation::kSUM);
    
    // 6. sqrt(variance + eps)
    auto std_dev = network->addUnary(*var_plus_eps->getOutput(0), nvinfer1::UnaryOperation::kSQRT);
    
    // 7. (x - mean) / sqrt(variance + eps)
    auto normalized = network->addElementWise(*x_sub_mean->getOutput(0), 
                                              *std_dev->getOutput(0), 
                                              nvinfer1::ElementWiseOperation::kDIV);
    
    // 8. 应用仿射变换: gamma * normalized + beta
    nvinfer1::Weights gamma_weights{nvinfer1::DataType::kFLOAT, gamma, dim};
    nvinfer1::Weights beta_weights{nvinfer1::DataType::kFLOAT, beta, dim};
    
    nvinfer1::Dims affine_dims;
    affine_dims.nbDims = 3;
    affine_dims.d[0] = 1;
    affine_dims.d[1] = 1;
    affine_dims.d[2] = dim;
    
    auto gamma_const = network->addConstant(affine_dims, gamma_weights);
    auto beta_const = network->addConstant(affine_dims, beta_weights);
    
    // gamma * normalized
    auto scaled = network->addElementWise(*normalized->getOutput(0), 
                                         *gamma_const->getOutput(0), 
                                         nvinfer1::ElementWiseOperation::kPROD);
    
    // scaled + beta
    auto output = network->addElementWise(*scaled->getOutput(0), 
                                         *beta_const->getOutput(0), 
                                         nvinfer1::ElementWiseOperation::kSUM);
    
    return output;
}

// Multi-Head Attention
nvinfer1::ITensor* MultiHeadAttention(nvinfer1::INetworkDefinition* network,
                                     std::map<std::string, nvinfer1::Weights>& weightMap,
                                     nvinfer1::ITensor& input,
                                     nvinfer1::ITensor& attention_mask,
                                     int dim, int num_heads,
                                     const std::string& lname) {
    int head_dim = dim / num_heads;
    
    // Q, K, V Linear transformations
    auto q_output = FullyConnected3D(network, input, dim,
                                     weightMap[lname + ".linear_layers.0.weight"], 
                                     weightMap[lname + ".linear_layers.0.bias"]);
    
    auto k_output = FullyConnected3D(network, input, dim,
                                     weightMap[lname + ".linear_layers.1.weight"], 
                                     weightMap[lname + ".linear_layers.1.bias"]);
    
    auto v_output = FullyConnected3D(network, input, dim,
                                     weightMap[lname + ".linear_layers.2.weight"], 
                                     weightMap[lname + ".linear_layers.2.bias"]);
    
    // Reshape Q, K, V: [B, N, D] -> [B, N, num_heads, head_dim]
    auto q_reshape = network->addShuffle(*q_output);
    q_reshape->setReshapeDimensions(nvinfer1::Dims4{Config::BATCH_SIZE, Config::MAX_SEQ_LEN, num_heads, head_dim});
    
    auto k_reshape = network->addShuffle(*k_output);
    k_reshape->setReshapeDimensions(nvinfer1::Dims4{Config::BATCH_SIZE, Config::MAX_SEQ_LEN, num_heads, head_dim});
    
    auto v_reshape = network->addShuffle(*v_output);
    v_reshape->setReshapeDimensions(nvinfer1::Dims4{Config::BATCH_SIZE, Config::MAX_SEQ_LEN, num_heads, head_dim});
    
    // Transpose: [B, N, num_heads, head_dim] -> [B, num_heads, N, head_dim]
    auto q_transpose = network->addShuffle(*q_reshape->getOutput(0));
    q_transpose->setFirstTranspose(nvinfer1::Permutation{0, 2, 1, 3});
    
    auto k_transpose = network->addShuffle(*k_reshape->getOutput(0));
    k_transpose->setFirstTranspose(nvinfer1::Permutation{0, 2, 1, 3});
    
    auto v_transpose = network->addShuffle(*v_reshape->getOutput(0));
    v_transpose->setFirstTranspose(nvinfer1::Permutation{0, 2, 1, 3});
    
    // Reshape for batch matrix multiplication: [B*num_heads, N, head_dim]
    auto q_final_reshape = network->addShuffle(*q_transpose->getOutput(0));
    q_final_reshape->setReshapeDimensions(nvinfer1::Dims3{Config::BATCH_SIZE * num_heads, Config::MAX_SEQ_LEN, head_dim});
    
    auto k_final_reshape = network->addShuffle(*k_transpose->getOutput(0));
    k_final_reshape->setReshapeDimensions(nvinfer1::Dims3{Config::BATCH_SIZE * num_heads, Config::MAX_SEQ_LEN, head_dim});
    
    auto v_final_reshape = network->addShuffle(*v_transpose->getOutput(0));
    v_final_reshape->setReshapeDimensions(nvinfer1::Dims3{Config::BATCH_SIZE * num_heads, Config::MAX_SEQ_LEN, head_dim});
    
    // Q @ K^T
    auto qk_matmul = network->addMatrixMultiply(*q_final_reshape->getOutput(0), 
                                               nvinfer1::MatrixOperation::kNONE,
                                               *k_final_reshape->getOutput(0), 
                                               nvinfer1::MatrixOperation::kTRANSPOSE);
    assert(qk_matmul);
    
    // Scale by sqrt(head_dim)
    float scale_val = 1.0f / sqrt(head_dim);
    auto scale_const = network->addConstant(nvinfer1::Dims3{1, 1, 1}, g_ConstantManager.getConstant(scale_val));
    auto scaled_qk = network->addElementWise(*qk_matmul->getOutput(0), *scale_const->getOutput(0),
                                            nvinfer1::ElementWiseOperation::kPROD);
    
    // Apply attention mask from external input (like ONNX)
    // Step 1: Convert attention_mask to float
    auto attention_mask_float = network->addIdentity(attention_mask);
    attention_mask_float->setOutputType(0, nvinfer1::DataType::kFLOAT);
    
    // Step 2: Convert mask to mask values: 1.0 -> 0.0 (valid), 0.0 -> -1e9 (padding)
    // mask_inv = 1.0 - attention_mask: 1.0 -> 0.0, 0.0 -> 1.0
    // 注意：必须使用静态存储，避免悬空指针
    static std::vector<float> ones_vec_static(Config::BATCH_SIZE * Config::MAX_SEQ_LEN, 1.0f);
    nvinfer1::Weights ones_weights{nvinfer1::DataType::kFLOAT, ones_vec_static.data(), 
                                   Config::BATCH_SIZE * Config::MAX_SEQ_LEN};
    auto ones_const = network->addConstant(nvinfer1::Dims2{Config::BATCH_SIZE, Config::MAX_SEQ_LEN}, ones_weights);
    
    auto mask_inv = network->addElementWise(*ones_const->getOutput(0), *attention_mask_float->getOutput(0),
                                          nvinfer1::ElementWiseOperation::kSUB);
    
    // mask_neg = mask_inv * -1e9: 0.0 -> 0.0, 1.0 -> -1e9
    // 注意：必须使用静态存储，避免悬空指针
    static std::vector<float> large_val_vec_static(Config::BATCH_SIZE * Config::MAX_SEQ_LEN, -1e9f);
    nvinfer1::Weights large_val_weights{nvinfer1::DataType::kFLOAT, large_val_vec_static.data(), 
                                       Config::BATCH_SIZE * Config::MAX_SEQ_LEN};
    auto large_val_const = network->addConstant(nvinfer1::Dims2{Config::BATCH_SIZE, Config::MAX_SEQ_LEN}, large_val_weights);
    
    // Element-wise multiply: mask_inv (1.0 for padding) * -1e9 = -1e9 for padding
    auto mask_neg = network->addElementWise(*mask_inv->getOutput(0), *large_val_const->getOutput(0),
                                           nvinfer1::ElementWiseOperation::kPROD);
    
    // Step 3: Expand mask to [B*num_heads, N, N] shape for attention scores
    // Reference: bert_standalone.py mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
    // This creates [B, 1, N, N] where each row is the same (mask[i,j] = mask[i,0] for all j)
    // In TensorRT, we need [B*num_heads, N, N] because scores are reshaped to [B*num_heads, N, N]
    
    // First, expand mask along batch*heads dimension: [B, N] -> [B*num_heads, N]
    // Use Gather to repeat each batch row num_heads times
    // 注意：必须使用静态存储，避免悬空指针
    static std::vector<int32_t> gather_indices_static;
    if (gather_indices_static.empty()) {
        gather_indices_static.reserve(Config::BATCH_SIZE * num_heads);
        for (int b = 0; b < Config::BATCH_SIZE; b++) {
            for (int h = 0; h < num_heads; h++) {
                gather_indices_static.push_back(b);
            }
        }
    }
    nvinfer1::Weights gather_indices_weights{nvinfer1::DataType::kINT32, gather_indices_static.data(), 
                                             Config::BATCH_SIZE * num_heads};
    
    // Create 1D indices tensor [B*num_heads] for Gather
    // TensorRT requires at least Dims2, so use [1, B*num_heads] then reshape if needed
    auto gather_indices_const = network->addConstant(nvinfer1::Dims2{1, Config::BATCH_SIZE * num_heads}, 
                                                     gather_indices_weights);
    
    // Reshape to [B*num_heads] for Gather (Gather expects 1D indices)
    auto indices_reshape = network->addShuffle(*gather_indices_const->getOutput(0));
    indices_reshape->setReshapeDimensions(nvinfer1::Dims2{Config::BATCH_SIZE * num_heads, 1});
    
    // Ensure mask_neg is [B, N] for Gather
    auto mask_for_gather = network->addShuffle(*mask_neg->getOutput(0));
    mask_for_gather->setReshapeDimensions(nvinfer1::Dims2{Config::BATCH_SIZE, Config::MAX_SEQ_LEN});
    
    // Gather: Input [B, N], Indices [B*num_heads, 1], axis=0, Output [B*num_heads, N]
    // Note: Gather will use the first dimension of indices, so [B*num_heads, 1] works correctly
    auto mask_gathered = network->addGather(*mask_for_gather->getOutput(0), 
                                           *indices_reshape->getOutput(0), 0);
    mask_gathered->setNbElementWiseDims(0);
    
    // Now expand mask to [B*num_heads, N, N]
    // Python: mask.unsqueeze(2).expand(-1, -1, N, -1) or mask.repeat(1, x.size(1), 1)
    // TensorRT: We need to create [B*num_heads, N, N] where mask[i,j,k] = mask[i,k] for all j
    // 
    // 参考 vit_builder-batch16-done.cpp 的 CLS token broadcast 方法
    // 使用 Concatenation 显式重复（最可靠的方法）
    
    // Step 1: Reshape [B*num_heads, N] to [B*num_heads, 1, N]
    auto mask_3d = network->addShuffle(*mask_gathered->getOutput(0));
    mask_3d->setReshapeDimensions(nvinfer1::Dims3{Config::BATCH_SIZE * num_heads, 1, Config::MAX_SEQ_LEN});
    
    // Step 2: 使用 Concatenation 重复 N 次，沿 dim=1 concat
    // [B*num_heads, 1, N] * N -> [B*num_heads, N, N]
    // 每一行 [1, N] 重复 N 次，得到 [N, N]，每行相同
    std::vector<nvinfer1::ITensor*> mask_rows;
    mask_rows.reserve(Config::MAX_SEQ_LEN);
    for (int i = 0; i < Config::MAX_SEQ_LEN; i++) {
        mask_rows.push_back(mask_3d->getOutput(0));  // 重复同一个 [B*heads, 1, N] tensor
    }
    
    // Concatenate along dimension 1: [B*num_heads, 1, N] * N -> [B*num_heads, N, N]
    // Result: mask_expanded[i, j, k] = mask_3d[i, 0, k] for all j
    auto mask_concat = network->addConcatenation(mask_rows.data(), Config::MAX_SEQ_LEN);
    mask_concat->setAxis(1);  // Concatenate along the query sequence dimension
    auto mask_expanded = mask_concat->getOutput(0);
    
    // Step 3: Apply mask to scaled_qk: add mask values (padding becomes -1e9)
    auto masked_qk = network->addElementWise(*scaled_qk->getOutput(0), *mask_expanded,
                                            nvinfer1::ElementWiseOperation::kSUM);
    
    // Softmax (on masked scores)
    auto softmax = network->addSoftMax(*masked_qk->getOutput(0));
    softmax->setAxes(1 << 2); // Softmax on last dimension
    
    // Attention @ V
    auto attn_v_matmul = network->addMatrixMultiply(*softmax->getOutput(0),
                                                   nvinfer1::MatrixOperation::kNONE,
                                                   *v_final_reshape->getOutput(0),
                                                   nvinfer1::MatrixOperation::kNONE);
    assert(attn_v_matmul);
    
    // Reshape back: [B*num_heads, N, head_dim] -> [B, num_heads, N, head_dim]
    auto attn_reshape = network->addShuffle(*attn_v_matmul->getOutput(0));
    attn_reshape->setReshapeDimensions(nvinfer1::Dims4{Config::BATCH_SIZE, num_heads, Config::MAX_SEQ_LEN, head_dim});
    
    // Transpose: [B, num_heads, N, head_dim] -> [B, N, num_heads, head_dim]
    auto attn_transpose = network->addShuffle(*attn_reshape->getOutput(0));
    attn_transpose->setFirstTranspose(nvinfer1::Permutation{0, 2, 1, 3});
    
    // Reshape: [B, N, num_heads, head_dim] -> [B, N, dim]
    auto attn_final_reshape = network->addShuffle(*attn_transpose->getOutput(0));
    attn_final_reshape->setReshapeDimensions(nvinfer1::Dims3{Config::BATCH_SIZE, Config::MAX_SEQ_LEN, dim});
    
    // Output projection
    auto out_proj = FullyConnected3D(network, *attn_final_reshape->getOutput(0), dim,
                                     weightMap[lname + ".output_linear.weight"],
                                     weightMap[lname + ".output_linear.bias"]);
    
    return out_proj;
}

// Feed Forward Network
nvinfer1::ITensor* FeedForward(nvinfer1::INetworkDefinition* network,
                               std::map<std::string, nvinfer1::Weights>& weightMap,
                               nvinfer1::ITensor& input,
                               int dim, int ffn_dim,
                               const std::string& lname) {
    // First linear layer
    auto fc1_output = FullyConnected3D(network, input, ffn_dim,
                                       weightMap[lname + ".w_1.weight"],
                                       weightMap[lname + ".w_1.bias"]);
    
    // GELU activation
    auto gelu_output = GELU(network, *fc1_output);
    assert(gelu_output);
    
    // Second linear layer
    auto fc2_output = FullyConnected3D(network, *gelu_output, dim,
                                       weightMap[lname + ".w_2.weight"],
                                       weightMap[lname + ".w_2.bias"]);
    
    return fc2_output;
}

// Transformer Block (Post-norm: BERT style)
nvinfer1::ITensor* TransformerBlock(nvinfer1::INetworkDefinition* network,
                                   std::map<std::string, nvinfer1::Weights>& weightMap,
                                   nvinfer1::ITensor& input,
                                   nvinfer1::ITensor& attention_mask,
                                   int dim, int num_heads, int ffn_dim,
                                   const std::string& lname) {
    // Multi-head attention
    auto attn = MultiHeadAttention(network, weightMap, input, attention_mask, dim, num_heads, lname + ".attention");
    
    // Dropout (skip in inference) + Residual connection 1
    auto residual1 = network->addElementWise(input, *attn, nvinfer1::ElementWiseOperation::kSUM);
    assert(residual1);
    
    // Layer norm 1 (post-norm)
    auto norm1 = LayerNorm(network, weightMap, *residual1->getOutput(0), dim, lname + ".input_sublayer.norm");
    
    // Feed Forward
    auto ffn = FeedForward(network, weightMap, *norm1->getOutput(0), dim, ffn_dim, lname + ".feed_forward");
    
    // Dropout (skip in inference) + Residual connection 2
    auto residual2 = network->addElementWise(*norm1->getOutput(0), *ffn, nvinfer1::ElementWiseOperation::kSUM);
    assert(residual2);
    
    // Layer norm 2 (post-norm)
    auto norm2 = LayerNorm(network, weightMap, *residual2->getOutput(0), dim, lname + ".output_sublayer.norm");
    
    return norm2->getOutput(0);
}

// ==================== BERT Network Builder Class ====================
class BERTBuilder {
private:
    nvinfer1::IBuilder* builder;
    nvinfer1::IBuilderConfig* config;
    nvinfer1::INetworkDefinition* network;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IRuntime* runtime;

    std::map<std::string, nvinfer1::Weights> weightMap;
    std::vector<void*> additionalMemory;  // Track additional allocated memory
    
    // 动态模型配置（根据 base/large 变化）
    int hidden_dim;
    int num_layers;
    int num_heads;
    int ffn_dim;

public:
    BERTBuilder() : builder(nullptr), config(nullptr), network(nullptr),
                    engine(nullptr), runtime(nullptr),
                    hidden_dim(768), num_layers(12), num_heads(12), ffn_dim(3072) {}

    ~BERTBuilder() {
        if (engine) engine->destroy();
        if (network) network->destroy();
        if (config) config->destroy();
        if (builder) builder->destroy();
        if (runtime) runtime->destroy();

        for (auto& mem : weightMap) {
            free((void*)(mem.second.values));
        }
        
        // Free additional allocated memory
        for (auto& mem : additionalMemory) {
            free(mem);
        }
    }

    void SetModelConfig(const std::string& model_type) {
        if (model_type == "large") {
            hidden_dim = 1024;
            num_layers = 24;
            num_heads = 16;
            ffn_dim = 4096;  // 4 * hidden_dim
        } else { // base
            hidden_dim = 768;
            num_layers = 12;
            num_heads = 12;
            ffn_dim = 3072;  // 4 * hidden_dim
        }
        std::cout << "Model: BERT-" << model_type << " (hidden_dim=" << hidden_dim 
                  << ", num_layers=" << num_layers << ", num_heads=" << num_heads << ")" << std::endl;
    }

    void Build(const std::string& wts_file, const std::string& model_type) {
        auto total_start = std::chrono::high_resolution_clock::now();
        
        weightMap = LoadWeights(wts_file);
        SetModelConfig(model_type);

        builder = nvinfer1::createInferBuilder(gLogger);
        network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

        std::cout << "\n=== Building BERT network layers ===" << std::endl;
        auto network_start = std::chrono::high_resolution_clock::now();
        BuildBERTNetwork();
        auto network_end = std::chrono::high_resolution_clock::now();
        auto network_duration = std::chrono::duration_cast<std::chrono::milliseconds>(network_end - network_start);
        std::cout << "Network construction time: " << network_duration.count() << " ms" << std::endl;

        config = builder->createBuilderConfig();
        config->setMaxWorkspaceSize(2ULL << 30);  // 2GB
        
        if (builder->platformHasFastFp16()) {
            std::cout << "✓ Enabling FP16 precision" << std::endl;
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        
        config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

        std::cout << "\n=== Optimizing and building engine ===" << std::endl;
        std::cout << "This may take several minutes..." << std::endl;
        auto build_start = std::chrono::high_resolution_clock::now();
        auto serialized_model = builder->buildSerializedNetwork(*network, *config);
        auto build_end = std::chrono::high_resolution_clock::now();
        auto build_duration = std::chrono::duration_cast<std::chrono::seconds>(build_end - build_start);
        std::cout << "Engine optimization & build time: " << build_duration.count() << " seconds" << std::endl;

        if (!serialized_model) {
            std::cerr << "ERROR: buildSerializedNetwork failed" << std::endl;
            return;
        }

        runtime = nvinfer1::createInferRuntime(gLogger);
        if (!runtime) {
            std::cerr << "ERROR: createInferRuntime failed" << std::endl;
            delete serialized_model;
            return;
        }

        engine = runtime->deserializeCudaEngine(serialized_model->data(), serialized_model->size());
        if (!engine) {
            std::cerr << "ERROR: deserializeCudaEngine failed" << std::endl;
            delete serialized_model;
            return;
        }

        delete serialized_model;
        
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);
        
        std::cout << "\n✓ Engine built successfully!" << std::endl;
        std::cout << "Total time: " << total_duration.count() << " seconds" << std::endl;
        std::cout << "Engine size: " << (engine->getDeviceMemorySize() / (1024.0 * 1024.0)) << " MB" << std::endl;
    }

    void Serialize(const std::string& engine_file) {
        if (!engine) {
            std::cerr << "ERROR: No engine to serialize" << std::endl;
            return;
        }

        std::cout << "\n=== Serializing engine ===" << std::endl;
        auto serialize_start = std::chrono::high_resolution_clock::now();
        
        auto serialized = engine->serialize();
        if (!serialized) {
            std::cerr << "ERROR: engine serialize failed" << std::endl;
            return;
        }

        std::ofstream out(engine_file, std::ios::binary);
        if (!out.is_open()) {
            std::cerr << "ERROR: Cannot open file for writing: " << engine_file << std::endl;
            delete serialized;
            return;
        }
        
        out.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());
        out.close();
        
        auto serialize_end = std::chrono::high_resolution_clock::now();
        auto serialize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(serialize_end - serialize_start);
        
        double size_mb = serialized->size() / (1024.0 * 1024.0);
        delete serialized;
        
        std::cout << "✓ Engine saved to: " << engine_file << std::endl;
        std::cout << "  File size: " << std::fixed << std::setprecision(2) << size_mb << " MB" << std::endl;
        std::cout << "  Serialization time: " << serialize_duration.count() << " ms" << std::endl;
    }

private:
    void BuildBERTNetwork() {
        std::cout << "Building BERT network..." << std::endl;

        // Input: token IDs [B, seq_len]
        auto input_ids = network->addInput(Config::INPUT_IDS_NAME, nvinfer1::DataType::kINT32,
                                          nvinfer1::Dims2{Config::BATCH_SIZE, Config::MAX_SEQ_LEN});
        
        // Input: attention mask [B, seq_len] (like ONNX)
        auto attention_mask = network->addInput(Config::ATTENTION_MASK_NAME, nvinfer1::DataType::kINT32,
                                                 nvinfer1::Dims2{Config::BATCH_SIZE, Config::MAX_SEQ_LEN});
        
        // Input: token type IDs [B, seq_len] (segment IDs, renamed to match ONNX)
        auto segment_ids = network->addInput(Config::SEGMENT_IDS_NAME, nvinfer1::DataType::kINT32,
                                            nvinfer1::Dims2{Config::BATCH_SIZE, Config::MAX_SEQ_LEN});
        
        // Embedding layers
        // Token Embedding
        nvinfer1::Dims token_emb_dims;
        token_emb_dims.nbDims = 2;
        token_emb_dims.d[0] = Config::VOCAB_SIZE;
        token_emb_dims.d[1] = hidden_dim;
        auto token_emb_weights = network->addConstant(token_emb_dims, weightMap["embedding.token.weight"]);
        auto token_gather = network->addGather(*token_emb_weights->getOutput(0), *input_ids, 0);
        token_gather->setNbElementWiseDims(0);
        
        // Position Embedding
        // Note: BERT has 512 position embeddings, but we only need MAX_SEQ_LEN
        // Need to extract the first MAX_SEQ_LEN rows from the full position embedding matrix
        nvinfer1::Weights full_pos_emb = weightMap["embedding.position.pe.weight"];
        size_t pos_emb_size = Config::MAX_SEQ_LEN * hidden_dim;
        float* pos_emb_data = reinterpret_cast<float*>(malloc(sizeof(float) * pos_emb_size));
        additionalMemory.push_back(pos_emb_data);  // Track for cleanup
        
        // Copy only the first MAX_SEQ_LEN rows
        const float* full_pos_data = reinterpret_cast<const float*>(full_pos_emb.values);
        for (int i = 0; i < Config::MAX_SEQ_LEN; i++) {
            for (int j = 0; j < hidden_dim; j++) {
                pos_emb_data[i * hidden_dim + j] = full_pos_data[i * hidden_dim + j];
            }
        }
        
        nvinfer1::Weights pos_emb_truncated{nvinfer1::DataType::kFLOAT, pos_emb_data, 
                                             static_cast<int64_t>(pos_emb_size)};
        nvinfer1::Dims pos_emb_dims;
        pos_emb_dims.nbDims = 2;
        pos_emb_dims.d[0] = Config::MAX_SEQ_LEN;
        pos_emb_dims.d[1] = hidden_dim;
        auto pos_emb_weights = network->addConstant(pos_emb_dims, pos_emb_truncated);
        
        // Create position indices [0, 1, 2, ..., seq_len-1] for all batches
        // 注意：必须使用 static 存储，避免悬空指针（参考 ViT 的实现）
        static std::vector<int32_t> position_ids_static;
        if (position_ids_static.empty()) {
            position_ids_static.resize(Config::BATCH_SIZE * Config::MAX_SEQ_LEN);
            for (int b = 0; b < Config::BATCH_SIZE; b++) {
                for (int i = 0; i < Config::MAX_SEQ_LEN; i++) {
                    position_ids_static[b * Config::MAX_SEQ_LEN + i] = i;
                }
            }
        }
        nvinfer1::Weights pos_ids_weights{nvinfer1::DataType::kINT32, position_ids_static.data(), 
                                         Config::BATCH_SIZE * Config::MAX_SEQ_LEN};
        auto pos_ids_const = network->addConstant(nvinfer1::Dims2{Config::BATCH_SIZE, Config::MAX_SEQ_LEN}, 
                                                 pos_ids_weights);
        auto pos_gather = network->addGather(*pos_emb_weights->getOutput(0), *pos_ids_const->getOutput(0), 0);
        pos_gather->setNbElementWiseDims(0);
        
        // Segment Embedding
        nvinfer1::Dims seg_emb_dims;
        seg_emb_dims.nbDims = 2;
        seg_emb_dims.d[0] = 3;  // [0, 1, 2]
        seg_emb_dims.d[1] = hidden_dim;
        auto seg_emb_weights = network->addConstant(seg_emb_dims, weightMap["embedding.segment.weight"]);
        auto seg_gather = network->addGather(*seg_emb_weights->getOutput(0), *segment_ids, 0);
        seg_gather->setNbElementWiseDims(0);
        
        // Sum embeddings: token + position + segment
        auto emb_sum1 = network->addElementWise(*token_gather->getOutput(0), *pos_gather->getOutput(0),
                                               nvinfer1::ElementWiseOperation::kSUM);
        auto emb_sum = network->addElementWise(*emb_sum1->getOutput(0), *seg_gather->getOutput(0),
                                              nvinfer1::ElementWiseOperation::kSUM);
        
        // Embedding LayerNorm (using PyTorch's LayerNorm, not custom)
        // Note: BERT uses nn.LayerNorm, not the custom LayerNorm class
        float* emb_ln_weight = (float*)weightMap["embedding.layer_norm.weight"].values;
        float* emb_ln_bias = (float*)weightMap["embedding.layer_norm.bias"].values;
        
        const float eps = 1e-12f;
        uint32_t reduceAxes = 1 << 2;
        auto emb_mean = network->addReduce(*emb_sum->getOutput(0), nvinfer1::ReduceOperation::kAVG, reduceAxes, true);
        auto emb_sub_mean = network->addElementWise(*emb_sum->getOutput(0), *emb_mean->getOutput(0), 
                                                    nvinfer1::ElementWiseOperation::kSUB);
        auto emb_sub_mean_sq = network->addElementWise(*emb_sub_mean->getOutput(0), *emb_sub_mean->getOutput(0), 
                                                       nvinfer1::ElementWiseOperation::kPROD);
        auto emb_variance = network->addReduce(*emb_sub_mean_sq->getOutput(0), 
                                              nvinfer1::ReduceOperation::kAVG, reduceAxes, true);
        auto eps_const = network->addConstant(nvinfer1::Dims3{1, 1, 1}, g_ConstantManager.getConstant(eps));
        auto emb_var_eps = network->addElementWise(*emb_variance->getOutput(0), *eps_const->getOutput(0), 
                                                   nvinfer1::ElementWiseOperation::kSUM);
        auto emb_std = network->addUnary(*emb_var_eps->getOutput(0), nvinfer1::UnaryOperation::kSQRT);
        auto emb_normalized = network->addElementWise(*emb_sub_mean->getOutput(0), *emb_std->getOutput(0), 
                                                     nvinfer1::ElementWiseOperation::kDIV);
        
        nvinfer1::Weights emb_gamma_weights{nvinfer1::DataType::kFLOAT, emb_ln_weight, hidden_dim};
        nvinfer1::Weights emb_beta_weights{nvinfer1::DataType::kFLOAT, emb_ln_bias, hidden_dim};
        auto emb_gamma = network->addConstant(nvinfer1::Dims3{1, 1, hidden_dim}, emb_gamma_weights);
        auto emb_beta = network->addConstant(nvinfer1::Dims3{1, 1, hidden_dim}, emb_beta_weights);
        auto emb_scaled = network->addElementWise(*emb_normalized->getOutput(0), *emb_gamma->getOutput(0), 
                                                 nvinfer1::ElementWiseOperation::kPROD);
        auto emb_output = network->addElementWise(*emb_scaled->getOutput(0), *emb_beta->getOutput(0), 
                                                 nvinfer1::ElementWiseOperation::kSUM);
        
        // Transformer blocks
        // Pass attention_mask as input (like ONNX)
        auto x = emb_output->getOutput(0);
        for (int i = 0; i < num_layers; i++) {
            x = TransformerBlock(network, weightMap, *x, *attention_mask, 
                                hidden_dim, num_heads, ffn_dim,
                                "transformer_blocks." + std::to_string(i));
        }

        // Output: [B, seq_len, hidden_dim]
        x->setName(Config::OUTPUT_NAME);
        network->markOutput(*x);

        std::cout << "BERT network built successfully! Output: ["
                  << Config::BATCH_SIZE << ", " << Config::MAX_SEQ_LEN << ", " 
                  << hidden_dim << "]" << std::endl;
    }
};

// ==================== Main Function ====================
int main(int argc, char** argv) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "       BERT TensorRT Engine Builder (Batch " << Config::BATCH_SIZE << ")" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  • Batch size: " << Config::BATCH_SIZE << std::endl;
    std::cout << "  • Max sequence length: " << Config::MAX_SEQ_LEN << std::endl;
    std::cout << "  • Vocabulary size: " << Config::VOCAB_SIZE << std::endl;
    std::cout << std::string(70, '=') << "\n" << std::endl;

    if (argc < 4) {
        std::cout << "Usage:" << std::endl;
        std::cout << "  ./bert_builder <wts_file> <engine_file> <model_type>" << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  ./bert_builder bert-base-uncased.wts bert_base_batch16.engine base" << std::endl;
        std::cout << "  ./bert_builder bert-large-uncased.wts bert_large_batch16.engine large" << std::endl;
        std::cout << "\nModel types:" << std::endl;
        std::cout << "  • base  - BERT-Base (768 dim, 12 layers, 12 heads)" << std::endl;
        std::cout << "  • large - BERT-Large (1024 dim, 24 layers, 16 heads)" << std::endl;
        return -1;
    }

    cudaSetDevice(Config::GPU_ID);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, Config::GPU_ID);
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total Memory: " << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) 
              << " GB\n" << std::endl;

    BERTBuilder builder;
    std::cout << "=== Starting Engine Build Process ===" << std::endl;
    builder.Build(argv[1], argv[3]);
    builder.Serialize(argv[2]);
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "✓ Build process completed successfully!" << std::endl;
    std::cout << std::string(70, '=') << "\n" << std::endl;

    return 0;
}

