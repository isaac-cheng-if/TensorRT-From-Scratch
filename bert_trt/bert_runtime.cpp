/*
 * BERT Runtime Inference
 * Handles engine deserialization, inference execution, and sentence similarity analysis
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
#include <sstream>
#include <numeric>
#include <algorithm>
#include <string>
#include <cctype>
#include <cstring>
#include <limits>

#include "NvInfer.h"
#include "cuda_runtime_api.h"

// ==================== Configuration ====================
struct BERTConfig {
    int BATCH_SIZE = 16;
    int MAX_SEQ_LEN = 128;
    int HIDDEN_DIM = 768;  // Will be auto-detected from engine
    const char* INPUT_IDS_NAME = "input_ids";
    const char* ATTENTION_MASK_NAME = "attention_mask";
    const char* SEGMENT_IDS_NAME = "token_type_ids";
    const char* OUTPUT_NAME = "output";
    int GPU_ID = 0;

    void DetectFromEngine(nvinfer1::ICudaEngine* engine) {
        // Auto-detect HIDDEN_DIM from engine output dimensions
        for (int i = 0; i < engine->getNbBindings(); i++) {
            if (!engine->bindingIsInput(i)) {
                // Found an output binding
                nvinfer1::Dims dims = engine->getBindingDimensions(i);
                // Output format: [batch, seq_len, hidden_dim]
                if (dims.nbDims == 3 && dims.d[2] > 0) {
                    HIDDEN_DIM = dims.d[2];
                    std::cout << "🔍 Auto-detected HIDDEN_DIM = " << HIDDEN_DIM << " from engine" << std::endl;
                    std::string model_type = (HIDDEN_DIM == 1024) ? "BERT-Large" : "BERT-Base";
                    std::cout << "   Model type: " << model_type << std::endl;
                    break;
                }
            }
        }
    }

    void Print() const {
        std::cout << "Configuration:" << std::endl;
        std::cout << "  • Batch size: " << BATCH_SIZE << std::endl;
        std::cout << "  • Max sequence length: " << MAX_SEQ_LEN << std::endl;
        std::cout << "  • Hidden dimension: " << HIDDEN_DIM << std::endl;
    }
};

BERTConfig Config;

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

// ==================== Performance Statistics ====================
struct PerfStats {
    int iterations;
    double avg_time;
    double min_time;
    double max_time;
    double total_time;
    double throughput;

    PerfStats() : iterations(0), avg_time(0), min_time(0), max_time(0),
                  total_time(0), throughput(0) {}

    void print(const std::string& title) const {
        std::cout << "\n=== " << title << " ===\n";
        std::cout << "Iterations: " << iterations << "\n";
        std::cout << "Average time: " << std::fixed << std::setprecision(3) << avg_time << " ms\n";
        std::cout << "Min time: " << std::fixed << std::setprecision(3) << min_time << " ms\n";
        std::cout << "Max time: " << std::fixed << std::setprecision(3) << max_time << " ms\n";
        std::cout << "Total time: " << std::fixed << std::setprecision(3) << total_time << " ms\n";
        std::cout << "Throughput: " << std::fixed << std::setprecision(2) << throughput << " sentences/sec\n";
        double batches_per_sec = throughput / Config.BATCH_SIZE;
        std::cout << "Batch Throughput: " << std::fixed << std::setprecision(2)
                  << batches_per_sec << " batches/sec (batch size: " << Config.BATCH_SIZE << ")\n";

        double tokens_per_batch = static_cast<double>(Config.BATCH_SIZE) * Config.MAX_SEQ_LEN;
        double tokens_per_sec = tokens_per_batch * batches_per_sec;
        std::cout << "Token Throughput: " << std::fixed << std::setprecision(2)
                << tokens_per_sec << " tokens/sec (" << tokens_per_batch << " tokens/batch)\n";

        std::cout << "====================================\n";
    }
};

// ==================== Binary Data Loader ====================
struct PreprocessedData {
    std::vector<std::vector<int>> input_ids_batch;
    std::vector<std::vector<int>> segment_ids_batch;
    std::vector<std::vector<int>> attention_masks_batch;
    std::vector<std::string> sentences;
    int batch_size;
    int max_seq_len;
};

bool LoadBinaryData(const std::string& data_prefix, PreprocessedData& data) {
    std::cout << "📥 加载预处理数据: " << data_prefix << std::endl;

    // 文件路径
    std::string input_ids_file = data_prefix + "_input_ids.bin";
    std::string segment_ids_file = data_prefix + "_segment_ids.bin";
    std::string attention_masks_file = data_prefix + "_attention_masks.bin";
    std::string metadata_file = data_prefix + "_metadata.txt";

    // 读取元数据获取形状信息
    std::ifstream meta(metadata_file);
    if (!meta.is_open()) {
        std::cerr << "❌ 无法打开元数据文件: " << metadata_file << std::endl;
        return false;
    }

    std::string line;
    int num_sentences = 0;
    data.batch_size = Config.BATCH_SIZE;
    data.max_seq_len = Config.MAX_SEQ_LEN;

    // 解析元数据
    bool in_sentences = false;
    while (std::getline(meta, line)) {
        if (line.find("句子数量:") != std::string::npos) {
            size_t pos = line.find(":");
            if (pos != std::string::npos) {
                num_sentences = std::stoi(line.substr(pos + 1));
            }
        }
        if (line.find("句子列表:") != std::string::npos) {
            in_sentences = true;
            std::getline(meta, line); // Skip separator
            continue;
        }
        if (in_sentences && !line.empty() && line[0] == '[') {
            // 提取句子
            size_t start = line.find("]") + 2;
            if (start < line.length()) {
                data.sentences.push_back(line.substr(start));
            }
        }
    }
    meta.close();

    std::cout << "  批次大小: " << data.batch_size << std::endl;
    std::cout << "  序列长度: " << data.max_seq_len << std::endl;
    std::cout << "  句子数量: " << data.sentences.size() << std::endl;

    // 计算总元素数
    size_t total_elements = data.batch_size * data.max_seq_len;
    size_t total_bytes = total_elements * sizeof(int32_t);

    // 读取 input_ids
    std::ifstream input_ids_f(input_ids_file, std::ios::binary);
    if (!input_ids_f.is_open()) {
        std::cerr << "❌ 无法打开文件: " << input_ids_file << std::endl;
        return false;
    }

    std::vector<int32_t> input_ids_flat(total_elements);
    input_ids_f.read(reinterpret_cast<char*>(input_ids_flat.data()), total_bytes);
    input_ids_f.close();

    // 读取 segment_ids
    std::ifstream segment_ids_f(segment_ids_file, std::ios::binary);
    if (!segment_ids_f.is_open()) {
        std::cerr << "❌ 无法打开文件: " << segment_ids_file << std::endl;
        return false;
    }

    std::vector<int32_t> segment_ids_flat(total_elements);
    segment_ids_f.read(reinterpret_cast<char*>(segment_ids_flat.data()), total_bytes);
    segment_ids_f.close();

    // 读取 attention_masks
    std::ifstream attention_masks_f(attention_masks_file, std::ios::binary);
    if (!attention_masks_f.is_open()) {
        std::cerr << "❌ 无法打开文件: " << attention_masks_file << std::endl;
        return false;
    }

    std::vector<int32_t> attention_masks_flat(total_elements);
    attention_masks_f.read(reinterpret_cast<char*>(attention_masks_flat.data()), total_bytes);
    attention_masks_f.close();

    // 重塑为批次
    data.input_ids_batch.resize(data.batch_size);
    data.segment_ids_batch.resize(data.batch_size);
    data.attention_masks_batch.resize(data.batch_size);

    for (int b = 0; b < data.batch_size; b++) {
        data.input_ids_batch[b].resize(data.max_seq_len);
        data.segment_ids_batch[b].resize(data.max_seq_len);
        data.attention_masks_batch[b].resize(data.max_seq_len);

        for (int i = 0; i < data.max_seq_len; i++) {
            int idx = b * data.max_seq_len + i;
            data.input_ids_batch[b][i] = input_ids_flat[idx];
            data.segment_ids_batch[b][i] = segment_ids_flat[idx];
            data.attention_masks_batch[b][i] = attention_masks_flat[idx];
        }
    }

    std::cout << "✅ 数据加载完成" << std::endl;
    std::cout << "  Input IDs: " << total_bytes << " bytes" << std::endl;
    std::cout << "  Segment IDs: " << total_bytes << " bytes" << std::endl;
    std::cout << "  Attention Masks: " << total_bytes << " bytes" << std::endl;

    return true;
}

// ==================== BERT Runtime Class ====================
class BERTRuntime {
private:
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    
    // 绑定索引缓存（避免每次推理都查找）
    int input_ids_idx;
    int attention_mask_idx;
    int token_type_ids_idx;
    int output_idx;  // last_hidden_state
    int pooler_output_idx;  // pooler_output (ONNX 第二个输出，可选)

    void* gpu_input_ids_buffer;
    void* gpu_attention_mask_buffer;
    void* gpu_segment_ids_buffer;
    void* gpu_output_buffer;  // last_hidden_state
    void* gpu_pooler_output_buffer;  // pooler_output (ONNX 第二个输出，可选)
    void* cpu_output_buffer;
    
    int input_ids_size;
    int attention_mask_size;
    int segment_ids_size;
    int output_size;
    int pooler_output_size;

    cudaStream_t stream;

public:
    BERTRuntime() : runtime(nullptr), engine(nullptr), context(nullptr),
                    gpu_input_ids_buffer(nullptr), gpu_attention_mask_buffer(nullptr),
                    gpu_segment_ids_buffer(nullptr), gpu_output_buffer(nullptr),
                    gpu_pooler_output_buffer(nullptr), cpu_output_buffer(nullptr),
                    input_ids_size(0), attention_mask_size(0), segment_ids_size(0), 
                    output_size(0), pooler_output_size(0), stream(nullptr),
                    input_ids_idx(-1), attention_mask_idx(-1), 
                    token_type_ids_idx(-1), output_idx(-1), pooler_output_idx(-1) {}

    ~BERTRuntime() {
        if (stream) cudaStreamDestroy(stream);
        if (cpu_output_buffer) free(cpu_output_buffer);
        if (gpu_pooler_output_buffer) cudaFree(gpu_pooler_output_buffer);
        if (gpu_output_buffer) cudaFree(gpu_output_buffer);
        if (gpu_segment_ids_buffer) cudaFree(gpu_segment_ids_buffer);
        if (gpu_attention_mask_buffer) cudaFree(gpu_attention_mask_buffer);
        if (gpu_input_ids_buffer) cudaFree(gpu_input_ids_buffer);
        if (context) context->destroy();
        if (engine) engine->destroy();
        if (runtime) runtime->destroy();
    }

    bool Deserialize(const std::string& engine_file) {
        std::cout << "📥 加载 TensorRT engine: " << engine_file << std::endl;

        std::ifstream file(engine_file, std::ios::binary);
        if (!file.good()) {
            std::cerr << "❌ 无法读取 engine 文件: " << engine_file << std::endl;
            return false;
        }

        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();

        runtime = nvinfer1::createInferRuntime(gLogger);
        if (!runtime) {
            std::cerr << "❌ createInferRuntime 失败" << std::endl;
            return false;
        }

        engine = runtime->deserializeCudaEngine(engine_data.data(), size);
        if (!engine) {
            std::cerr << "❌ deserializeCudaEngine 失败" << std::endl;
            return false;
        }

        
        // Auto-detect configuration from engine
        Config.DetectFromEngine(engine);
        context = engine->createExecutionContext();
        if (!context) {
            std::cerr << "❌ createExecutionContext 失败" << std::endl;
            return false;
        }

        // 检查引擎的输入输出信息（兼容 ONNX 和自定义 builder）
        int num_inputs = 0;
        int num_outputs = 0;
        for (int i = 0; i < engine->getNbBindings(); i++) {
            if (engine->bindingIsInput(i)) {
                num_inputs++;
            } else {
                num_outputs++;
            }
        }

        std::cout << "📊 Engine 信息:" << std::endl;
        std::cout << "  输入数量: " << num_inputs << std::endl;
        std::cout << "  输出数量: " << num_outputs << std::endl;
        
        // 打印所有绑定信息
        for (int i = 0; i < engine->getNbBindings(); i++) {
            const char* name = engine->getBindingName(i);
            bool is_input = engine->bindingIsInput(i);
            nvinfer1::Dims dims = engine->getBindingDimensions(i);
            std::cout << "  [" << i << "] " << (is_input ? "输入" : "输出") 
                      << ": " << (name ? name : "unknown") << " [";
            for (int j = 0; j < dims.nbDims; j++) {
                std::cout << dims.d[j];
                if (j < dims.nbDims - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }

        // Allocate GPU memory
        input_ids_size = Config.BATCH_SIZE * Config.MAX_SEQ_LEN * sizeof(int32_t);
        attention_mask_size = Config.BATCH_SIZE * Config.MAX_SEQ_LEN * sizeof(int32_t);
        segment_ids_size = Config.BATCH_SIZE * Config.MAX_SEQ_LEN * sizeof(int32_t);
        
        // ONNX 输出: last_hidden_state [B, N, D] 和 pooler_output [B, D]
        // 自定义 builder 输出: output [B, N, D]
        // 我们只需要 last_hidden_state (第一个输出)，但需要为 pooler_output 也分配缓冲区
        output_size = Config.BATCH_SIZE * Config.MAX_SEQ_LEN * Config.HIDDEN_DIM * sizeof(float);
        pooler_output_size = Config.BATCH_SIZE * Config.HIDDEN_DIM * sizeof(float);
        
        CUDA_CHECK(cudaMalloc(&gpu_input_ids_buffer, input_ids_size));
        CUDA_CHECK(cudaMalloc(&gpu_attention_mask_buffer, attention_mask_size));
        CUDA_CHECK(cudaMalloc(&gpu_segment_ids_buffer, segment_ids_size));
        CUDA_CHECK(cudaMalloc(&gpu_output_buffer, output_size));
        // 为 pooler_output 分配缓冲区（即使我们不使用它）
        CUDA_CHECK(cudaMalloc(&gpu_pooler_output_buffer, pooler_output_size));
        cpu_output_buffer = malloc(output_size);
        
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        // 查找并缓存绑定索引（兼容 ONNX 和自定义 builder）
        input_ids_idx = -1;
        attention_mask_idx = -1;
        token_type_ids_idx = -1;
        output_idx = -1;
        pooler_output_idx = -1;
        
        for (int i = 0; i < engine->getNbBindings(); i++) {
            const char* name = engine->getBindingName(i);
            if (!name) continue;
            
            if (engine->bindingIsInput(i)) {
                if (strcmp(name, Config.INPUT_IDS_NAME) == 0) input_ids_idx = i;
                else if (strcmp(name, Config.ATTENTION_MASK_NAME) == 0) attention_mask_idx = i;
                else if (strcmp(name, Config.SEGMENT_IDS_NAME) == 0) token_type_ids_idx = i;
            } else {
                // 输出: 优先使用 last_hidden_state (ONNX)，否则使用第一个输出
                if (strcmp(name, "last_hidden_state") == 0 || strcmp(name, Config.OUTPUT_NAME) == 0) {
                    if (output_idx == -1) output_idx = i;
                } else if (strcmp(name, "pooler_output") == 0) {
                    pooler_output_idx = i;
                } else if (output_idx == -1 && i >= 3) {
                    output_idx = i;
                }
            }
        }
        
        // 如果找不到名称匹配，使用默认顺序（假设标准顺序）
        if (input_ids_idx == -1) input_ids_idx = 0;
        if (attention_mask_idx == -1) attention_mask_idx = 1;
        if (token_type_ids_idx == -1) token_type_ids_idx = 2;
        if (output_idx == -1) {
            // 找到第一个输出绑定
            for (int i = 0; i < engine->getNbBindings(); i++) {
                if (!engine->bindingIsInput(i)) {
                    output_idx = i;
                    break;
                }
            }
        }
        // 如果找不到 pooler_output，设置为 -1（表示不存在）
        if (pooler_output_idx == -1) {
            // 查找第二个输出（如果有）
            int output_count = 0;
            for (int i = 0; i < engine->getNbBindings(); i++) {
                if (!engine->bindingIsInput(i)) {
                    output_count++;
                    if (output_count == 2 && i != output_idx) {
                        pooler_output_idx = i;
                        break;
                    }
                }
            }
        }
        
        std::cout << "\n✅ Engine 加载成功!" << std::endl;
        std::cout << "  绑定索引: input_ids=" << input_ids_idx 
                  << ", attention_mask=" << attention_mask_idx
                  << ", token_type_ids=" << token_type_ids_idx
                  << ", output=" << output_idx;
        if (pooler_output_idx != -1) {
            std::cout << ", pooler_output=" << pooler_output_idx;
        }
        std::cout << std::endl;
        std::cout << "  Input IDs 大小: " << input_ids_size << " bytes" << std::endl;
        std::cout << "  Attention Mask 大小: " << attention_mask_size << " bytes" << std::endl;
        std::cout << "  Token Type IDs 大小: " << segment_ids_size << " bytes" << std::endl;
        std::cout << "  输出大小 (last_hidden_state): " << output_size << " bytes" << std::endl;
        if (pooler_output_idx != -1) {
            std::cout << "  输出大小 (pooler_output): " << pooler_output_size << " bytes" << std::endl;
        }

        return true;
    }

    bool InferBatch(const std::vector<std::vector<int>>& input_ids_batch,
                   const std::vector<std::vector<int>>& attention_mask_batch,
                   const std::vector<std::vector<int>>& segment_ids_batch,
                   std::vector<std::vector<std::vector<float>>>& outputs) {
        
        if (input_ids_batch.size() != Config.BATCH_SIZE || 
            attention_mask_batch.size() != Config.BATCH_SIZE ||
            segment_ids_batch.size() != Config.BATCH_SIZE) {
            std::cerr << "ERROR: Batch size must be " << Config.BATCH_SIZE << std::endl;
            return false;
        }
        
        // Prepare input data
        std::vector<int32_t> input_ids_data(Config.BATCH_SIZE * Config.MAX_SEQ_LEN);
        std::vector<int32_t> attention_mask_data(Config.BATCH_SIZE * Config.MAX_SEQ_LEN);
        std::vector<int32_t> segment_ids_data(Config.BATCH_SIZE * Config.MAX_SEQ_LEN);
        
        for (int b = 0; b < Config.BATCH_SIZE; b++) {
            for (int i = 0; i < Config.MAX_SEQ_LEN; i++) {
                input_ids_data[b * Config.MAX_SEQ_LEN + i] = input_ids_batch[b][i];
                attention_mask_data[b * Config.MAX_SEQ_LEN + i] = attention_mask_batch[b][i];
                segment_ids_data[b * Config.MAX_SEQ_LEN + i] = segment_ids_batch[b][i];
            }
        }
        
        // Copy to GPU
        CUDA_CHECK(cudaMemcpyAsync(gpu_input_ids_buffer, input_ids_data.data(), input_ids_size, 
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_attention_mask_buffer, attention_mask_data.data(), attention_mask_size, 
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_segment_ids_buffer, segment_ids_data.data(), segment_ids_size, 
                                   cudaMemcpyHostToDevice, stream));
        
        // Run inference
        // executeV2 需要按照引擎绑定索引顺序传递缓冲区
        // 使用缓存的绑定索引（在 Deserialize 中已设置）
        
        // 创建 bindings 数组（按引擎绑定索引顺序）
        // executeV2 要求所有绑定都有有效的缓冲区指针（即使我们不使用某些输出）
        std::vector<void*> bindings(engine->getNbBindings(), nullptr);
        bindings[input_ids_idx] = gpu_input_ids_buffer;
        bindings[attention_mask_idx] = gpu_attention_mask_buffer;
        bindings[token_type_ids_idx] = gpu_segment_ids_buffer;
        bindings[output_idx] = gpu_output_buffer;
        if (pooler_output_idx != -1) {
            bindings[pooler_output_idx] = gpu_pooler_output_buffer;
        }
        
        bool status = context->executeV2(bindings.data());
        if (!status) {
            std::cerr << "ERROR: executeV2 failed" << std::endl;
            return false;
        }

        // Copy result back to CPU
        CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_output_buffer, output_size,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Convert to vector format [batch][seq_len][hidden_dim]
        float* output_ptr = static_cast<float*>(cpu_output_buffer);
        outputs.resize(Config.BATCH_SIZE);
        for (int b = 0; b < Config.BATCH_SIZE; b++) {
            outputs[b].resize(Config.MAX_SEQ_LEN);
            for (int s = 0; s < Config.MAX_SEQ_LEN; s++) {
                outputs[b][s].resize(Config.HIDDEN_DIM);
                int offset = (b * Config.MAX_SEQ_LEN + s) * Config.HIDDEN_DIM;
                for (int h = 0; h < Config.HIDDEN_DIM; h++) {
                    outputs[b][s][h] = output_ptr[offset + h];
                }
            }
        }

        return true;
    }

    std::vector<float> GetCLSEmbedding(const std::vector<std::vector<float>>& sequence_output) {
        // Extract [CLS] token embedding (first token)
        return sequence_output[0];
    }

    std::vector<float> GetMeanPooling(const std::vector<std::vector<float>>& sequence_output,
                                     const std::vector<int>& input_ids) {
        // Mean pooling over non-padding tokens
        std::vector<float> mean_embedding(Config.HIDDEN_DIM, 0.0f);
        int valid_tokens = 0;

        for (int i = 0; i < Config.MAX_SEQ_LEN; i++) {
            if (input_ids[i] != 0) { // Not padding
                for (int h = 0; h < Config.HIDDEN_DIM; h++) {
                    mean_embedding[h] += sequence_output[i][h];
                }
                valid_tokens++;
            }
        }

        if (valid_tokens > 0) {
            for (int h = 0; h < Config.HIDDEN_DIM; h++) {
                mean_embedding[h] /= valid_tokens;
            }
        }

        return mean_embedding;
    }

    float CosineSimilarity(const std::vector<float>& v1, const std::vector<float>& v2) {
        float dot_product = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;

        for (size_t i = 0; i < v1.size(); i++) {
            dot_product += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }

        if (norm1 == 0.0f || norm2 == 0.0f) return 0.0f;

        return dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
    }

    float EuclideanDistance(const std::vector<float>& v1, const std::vector<float>& v2) {
        float sum = 0.0f;
        for (size_t i = 0; i < v1.size(); i++) {
            float diff = v1[i] - v2[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    PerfStats PerformanceTest(const std::vector<std::vector<int>>& input_ids_batch,
                            const std::vector<std::vector<int>>& attention_mask_batch,
                            const std::vector<std::vector<int>>& segment_ids_batch,
                            int iterations = 100) {
        PerfStats stats;
        stats.iterations = iterations;
        
        if (input_ids_batch.size() != Config.BATCH_SIZE ||
            attention_mask_batch.size() != Config.BATCH_SIZE ||
            segment_ids_batch.size() != Config.BATCH_SIZE) {
            std::cerr << "ERROR: Batch size must be " << Config.BATCH_SIZE << std::endl;
            return stats;
        }
        
        std::vector<double> times;
        times.reserve(iterations);
        
        // Warmup
        std::vector<std::vector<std::vector<float>>> dummy_outputs;
        for (int i = 0; i < 10; i++) {
            InferBatch(input_ids_batch, attention_mask_batch, segment_ids_batch, dummy_outputs);
        }
        
        // Prepare input data once
        std::vector<int32_t> input_ids_data(Config.BATCH_SIZE * Config.MAX_SEQ_LEN);
        std::vector<int32_t> attention_mask_data(Config.BATCH_SIZE * Config.MAX_SEQ_LEN);
        std::vector<int32_t> segment_ids_data(Config.BATCH_SIZE * Config.MAX_SEQ_LEN);
        
        for (int b = 0; b < Config.BATCH_SIZE; b++) {
            for (int i = 0; i < Config.MAX_SEQ_LEN; i++) {
                input_ids_data[b * Config.MAX_SEQ_LEN + i] = input_ids_batch[b][i];
                attention_mask_data[b * Config.MAX_SEQ_LEN + i] = attention_mask_batch[b][i];
                segment_ids_data[b * Config.MAX_SEQ_LEN + i] = segment_ids_batch[b][i];
            }
        }
        
        // Performance test with CUDA events
        cudaEvent_t start_event, end_event;
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&end_event));
        
        for (int i = 0; i < iterations; i++) {
            // Copy to GPU
            CUDA_CHECK(cudaMemcpyAsync(gpu_input_ids_buffer, input_ids_data.data(), input_ids_size,
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(gpu_attention_mask_buffer, attention_mask_data.data(), attention_mask_size,
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(gpu_segment_ids_buffer, segment_ids_data.data(), segment_ids_size,
                                       cudaMemcpyHostToDevice, stream));
            
            // Start timing
            CUDA_CHECK(cudaEventRecord(start_event, stream));
            
            // Run inference (使用缓存的绑定索引)
            std::vector<void*> perf_bindings(engine->getNbBindings(), nullptr);
            perf_bindings[input_ids_idx] = gpu_input_ids_buffer;
            perf_bindings[attention_mask_idx] = gpu_attention_mask_buffer;
            perf_bindings[token_type_ids_idx] = gpu_segment_ids_buffer;
            perf_bindings[output_idx] = gpu_output_buffer;
            if (pooler_output_idx != -1) {
                perf_bindings[pooler_output_idx] = gpu_pooler_output_buffer;
            }
            bool status = context->executeV2(perf_bindings.data());
            if (!status) {
                std::cerr << "ERROR: executeV2 failed" << std::endl;
                return stats;
            }

            // End timing
            CUDA_CHECK(cudaEventRecord(end_event, stream));
            CUDA_CHECK(cudaEventSynchronize(end_event));

            float gpu_time_ms;
            CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start_event, end_event));
            times.push_back(gpu_time_ms);

            // Copy result back (not timed)
            CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_output_buffer, output_size,
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(end_event));

        // Calculate statistics
        stats.total_time = std::accumulate(times.begin(), times.end(), 0.0);
        stats.avg_time = stats.total_time / iterations;
        stats.min_time = *std::min_element(times.begin(), times.end());
        stats.max_time = *std::max_element(times.begin(), times.end());
        stats.throughput = (Config.BATCH_SIZE * 1000.0) / stats.avg_time;

        return stats;
    }

};

// ==================== Main Function ====================
int main(int argc, char** argv) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "BERT Runtime Inference (Batch " << Config.BATCH_SIZE << ")" << std::endl;
    std::cout << "TensorRT Engine Inference" << std::endl;
    std::cout << std::string(70, '=') << "\n" << std::endl;

    if (argc < 3) {
        std::cout << "用法:" << std::endl;
        std::cout << "  ./bert_runtime <engine_file> <data_prefix>" << std::endl;
        std::cout << "\n参数:" << std::endl;
        std::cout << "  engine_file  - TensorRT engine 文件路径" << std::endl;
        std::cout << "  data_prefix  - 预处理数据文件前缀" << std::endl;
        std::cout << "\n示例:" << std::endl;
        std::cout << "  ./bert_runtime bert_base_batch16.engine bert_input" << std::endl;
        std::cout << "\n准备数据:" << std::endl;
        std::cout << "  在有 Python 环境的机器上运行:" << std::endl;
        std::cout << "  python3 prepare_bert_input.py -i sentences.txt -o bert_input" << std::endl;
        std::cout << "\n  然后将生成的 .bin 文件传输到 Orin 设备" << std::endl;
        return -1;
    }

    cudaSetDevice(Config.GPU_ID);

    // 加载预处理数据
    std::cout << "步骤 1: 加载预处理数据" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    PreprocessedData data;
    if (!LoadBinaryData(argv[2], data)) {
        std::cerr << "❌ 加载数据失败" << std::endl;
        return -1;
    }

    std::cout << std::endl;

    // 加载 engine
    std::cout << "步骤 2: 加载 TensorRT Engine" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    BERTRuntime runtime;
    if (!runtime.Deserialize(argv[1])) {
        std::cerr << "❌ Engine 加载失败" << std::endl;
        return -1;
    }

    std::cout << std::endl;

    // 运行推理
    std::cout << "步骤 3: 运行推理" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    std::vector<std::vector<std::vector<float>>> outputs;
    if (runtime.InferBatch(data.input_ids_batch, data.attention_masks_batch, data.segment_ids_batch, outputs)) {
        std::cout << "✅ 推理成功!\n" << std::endl;

        // 提取 embeddings (使用 mean pooling)
        std::cout << "步骤 4: 提取句子嵌入" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        std::vector<std::vector<float>> embeddings;
        int num_valid = std::min(Config.BATCH_SIZE, (int)data.sentences.size());

        for (int b = 0; b < num_valid; b++) {
            auto embedding = runtime.GetMeanPooling(outputs[b], data.input_ids_batch[b]);
            embeddings.push_back(embedding);

            if (b < 3) {  // 只显示前3个
                std::cout << "  [" << b << "] \"" << data.sentences[b].substr(0, 50) << "...\"" << std::endl;
                std::cout << "      嵌入维度: " << embedding.size()
                          << ", 范围: [" << *std::min_element(embedding.begin(), embedding.end())
                          << ", " << *std::max_element(embedding.begin(), embedding.end()) << "]" << std::endl;
            }
        }

        std::cout << "✅ 共提取 " << embeddings.size() << " 个句子嵌入\n" << std::endl;

        // 计算相似度矩阵
        std::cout << "步骤 5: 计算相似度" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        std::cout << "\n句子列表:" << std::endl;
        for (int i = 0; i < num_valid; i++) {
            std::cout << "  [" << i << "] \"" << data.sentences[i] << "\"" << std::endl;
        }

        std::cout << "\n余弦相似度矩阵:" << std::endl;
        std::cout << "      ";
        for (int i = 0; i < num_valid; i++) {
            std::cout << "   [" << i << "]  ";
        }
        std::cout << std::endl;

        for (int i = 0; i < num_valid; i++) {
            std::cout << " [" << i << "]  ";
            for (int j = 0; j < num_valid; j++) {
                float cos_sim = runtime.CosineSimilarity(embeddings[i], embeddings[j]);
                std::cout << std::fixed << std::setprecision(4) << cos_sim << "  ";
            }
            std::cout << std::endl;
        }

        // 计算所有句子对的相似度
        std::vector<std::tuple<int, int, float, float>> similarities;
        for (int i = 0; i < num_valid; i++) {
            for (int j = i + 1; j < num_valid; j++) {
                float cos_sim = runtime.CosineSimilarity(embeddings[i], embeddings[j]);
                float euc_dist = runtime.EuclideanDistance(embeddings[i], embeddings[j]);
                similarities.push_back({i, j, cos_sim, euc_dist});
            }
        }

        // 按余弦相似度排序（从高到低）
        std::sort(similarities.begin(), similarities.end(),
                 [](const auto& a, const auto& b) { return std::get<2>(a) > std::get<2>(b); });

        // 最相似的句子对
        std::cout << "\n🎯 最相似的句子对:" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        for (int k = 0; k < std::min(5, (int)similarities.size()); k++) {
            int i = std::get<0>(similarities[k]);
            int j = std::get<1>(similarities[k]);
            float cos_sim = std::get<2>(similarities[k]);
            float euc_dist = std::get<3>(similarities[k]);

            std::cout << "\n" << (k+1) << ". 句子 " << i << " <-> 句子 " << j << std::endl;
            std::cout << "   A: \"" << data.sentences[i] << "\"" << std::endl;
            std::cout << "   B: \"" << data.sentences[j] << "\"" << std::endl;
            std::cout << "   余弦相似度: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
            std::cout << "   欧式距离:   " << std::fixed << std::setprecision(4) << euc_dist << std::endl;
        }

        std::cout << "\n" << std::string(70, '-') << std::endl;

        // 最不相似的句子对
        std::cout << "\n❌ 最不相似的句子对:" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        int start_idx = std::max(0, (int)similarities.size() - 5);
        for (int k = similarities.size() - 1; k >= start_idx && k >= 0; k--) {
            int i = std::get<0>(similarities[k]);
            int j = std::get<1>(similarities[k]);
            float cos_sim = std::get<2>(similarities[k]);
            float euc_dist = std::get<3>(similarities[k]);

            std::cout << "\n" << (similarities.size() - k) << ". 句子 " << i << " <-> 句子 " << j << std::endl;
            std::cout << "   A: \"" << data.sentences[i] << "\"" << std::endl;
            std::cout << "   B: \"" << data.sentences[j] << "\"" << std::endl;
            std::cout << "   余弦相似度: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
            std::cout << "   欧式距离:   " << std::fixed << std::setprecision(4) << euc_dist << std::endl;
        }

        // 统计信息
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "\n📈 统计信息:" << std::endl;

        float min_cos = std::get<2>(similarities.back());
        float max_cos = std::get<2>(similarities.front());
        float sum_cos = 0.0f;
        float min_euc = std::numeric_limits<float>::max();
        float max_euc = 0.0f;
        float sum_euc = 0.0f;

        for (const auto& sim : similarities) {
            sum_cos += std::get<2>(sim);
            float euc = std::get<3>(sim);
            sum_euc += euc;
            min_euc = std::min(min_euc, euc);
            max_euc = std::max(max_euc, euc);
        }

        float avg_cos = sum_cos / similarities.size();
        float avg_euc = sum_euc / similarities.size();

        std::cout << "  句子对总数:           " << similarities.size() << std::endl;
        std::cout << "  余弦相似度范围:       [" << std::fixed << std::setprecision(4)
                  << min_cos << ", " << max_cos << "]" << std::endl;
        std::cout << "  余弦相似度平均值:     " << std::fixed << std::setprecision(4) << avg_cos << std::endl;
        std::cout << "  欧式距离范围:         [" << std::fixed << std::setprecision(4)
                  << min_euc << ", " << max_euc << "]" << std::endl;
        std::cout << "  欧式距离平均值:       " << std::fixed << std::setprecision(4) << avg_euc << std::endl;

        std::cout << "\n💡 相似度解释:" << std::endl;
        std::cout << "  • 余弦相似度: 1.0 = 完全相同, 0.0 = 完全不相关, 接近1表示相似" << std::endl;
        std::cout << "  • 欧式距离:   < 6.0 = 相似, > 8.0 = 不相关, 越小越相似" << std::endl;

        // 性能测试
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "步骤 6: 性能基准测试" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        auto perf_stats = runtime.PerformanceTest(data.input_ids_batch, data.attention_masks_batch, data.segment_ids_batch, 100);
        perf_stats.print("GPU 性能 (批次大小 " + std::to_string(Config.BATCH_SIZE) + ")");
    } else {
        std::cerr << "❌ 推理失败" << std::endl;
        return -1;
    }

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "✅ 推理完成!" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    return 0;
}