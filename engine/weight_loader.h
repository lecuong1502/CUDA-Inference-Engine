// engine/weight_loader.h
#pragma once
#include <string>
#include <vector>
#include <unordered_map>

using namespace std;

struct TensorMeta;  // forward decl
 
class WeightLoader {
public:
    explicit WeightLoader(const string& safetensors_path);
    ~WeightLoader();
 
    // Load tensor by name into GPU memory (float32).
    // Caller does NOT own the pointer — freed by ~WeightLoader().
    float* load_to_gpu(const string& name);
 
    // Metadata queries
    vector<size_t> shape(const string& name) const;
    bool has(const string& name) const;
 
    // Debug: print all tensor names and shapes
    void print_tensors() const;
 
private:
    unordered_map<std::string, TensorMeta> metas_;
    vector<uint8_t> cpu_data_;
    vector<float*>  gpu_ptrs_;
    size_t data_offset_ = 0;
};