// engine/weight_loader.cpp
// Load GPT-2 weights from .safetensors format into GPU memory.
//
// safetensors format:
//   [8 bytes: header_len (uint64_le)]
//   [header_len bytes: JSON header]
//   [raw tensor data: contiguous float32/float16 blobs]
//
// We parse the JSON header to find each tensor's dtype, shape, and
// byte offset into the data region, then cudaMemcpy directly to GPU.

#include "weight_loader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <bits/stdc++.h>
using namespace std;

// ===================================================================
// Minimal JSON field extractor (no external JSON lib dependency)
// ===================================================================

static string extract_field(const string& json,
                                 const string& key) {
    string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == string::npos) return "";
    pos = json.find(':', pos) + 1;
    while (json[pos] == ' ') pos++;
    if (json[pos] == '"') {
        size_t end = json.find('"', pos + 1);
        return json.substr(pos + 1, end - pos - 1);
    }
    // Number or array
    size_t end = json.find_first_of(",}", pos);
    return json.substr(pos, end - pos);
}

static vector<size_t> parse_shape(const string& s) {
    vector<size_t> shape;
    string num;
    for (char c : s) {
        if (c >= '0' && c <= '9') num += c;
        else if (!num.empty()) { shape.push_back(stoul(num)); num.clear(); }
    }
    if (!num.empty()) shape.push_back(stoul(num));
    return shape;
}

static pair<size_t, size_t> parse_offsets(const string& s) {
    vector<size_t> nums;
    string num;
    for (char c : s) {
        if (c >= '0' && c <= '9') num += c;
        else if (!num.empty()) { nums.push_back(stoul(num)); num.clear(); }
    }
    if (!num.empty()) nums.push_back(stoul(num));
    assert(nums.size() == 2);
    return {nums[0], nums[1]};
}

// ===================================================================
// TensorMeta: parsed from safetensors JSON header
// ===================================================================

struct TensorMeta {
    string dtype;
    vector<size_t> shape;
    size_t data_start;
    size_t data_end;

    size_t num_elements() const {
        size_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
    size_t byte_size() const {
        size_t elem_bytes = (dtype == "F32") ? 4 : 2;  // F32 or F16
        return num_elements() * elem_bytes;
    }
};

// Parse all tensor metadata from safetensors JSON header
static unordered_map<string, TensorMeta>
parse_header(const string& header_json) {
    unordered_map<string, TensorMeta> metas;

    size_t pos = 0;
    while (pos < header_json.size()) {
        // Find next tensor name: "name": {
        size_t name_start = header_json.find('"', pos);
        if (name_start == string::npos) break;
        size_t name_end = header_json.find('"', name_start + 1);
        string name = header_json.substr(name_start + 1,
                                              name_end - name_start - 1);

        if (name == "__metadata__") { pos = name_end + 1; continue; }

        // Find the object { ... } for this tensor
        size_t obj_start = header_json.find('{', name_end);
        size_t obj_end   = header_json.find('}', obj_start);
        string obj  = header_json.substr(obj_start, obj_end - obj_start + 1);

        TensorMeta meta;
        meta.dtype = extract_field(obj, "dtype");

        string shape_str   = extract_field(obj, "shape");
        string offsets_str = extract_field(obj, "data_offsets");

        meta.shape = parse_shape(shape_str);
        auto [s, e] = parse_offsets(offsets_str);
        meta.data_start = s;
        meta.data_end   = e;

        metas[name] = meta;
        pos = obj_end + 1;
    }
    return metas;
}

// ===================================================================
// WeightLoader implementation
// ===================================================================

WeightLoader::WeightLoader(const string& path) {
    ifstream f(path, ios::binary);
    if (!f.is_open())
        throw runtime_error("Cannot open: " + path);

    // Read 8-byte header length
    uint64_t header_len;
    f.read(reinterpret_cast<char*>(&header_len), 8);

    // Read JSON header
    string header_json(header_len, '\0');
    f.read(header_json.data(), header_len);

    metas_ = parse_header(header_json);

    // Read all raw data into CPU buffer
    data_offset_ = 8 + header_len;
    f.seekg(0, ios::end);
    size_t file_size = f.tellg();
    size_t data_size = file_size - data_offset_;

    cpu_data_.resize(data_size);
    f.seekg(data_offset_);
    f.read(reinterpret_cast<char*>(cpu_data_.data()), data_size);

    cout << "[WeightLoader] Loaded " << path
              << " | tensors: " << metas_.size()
              << " | data: " << data_size / 1024 / 1024 << " MB\n";
}

float* WeightLoader::load_to_gpu(const string& name) {
    auto it = metas_.find(name);
    if (it == metas_.end())
        throw runtime_error("Tensor not found: " + name);

    const TensorMeta& meta = it->second;
    size_t n = meta.num_elements();

    float* d_ptr = nullptr;
    cudaMalloc(&d_ptr, n * sizeof(float));

    if (meta.dtype == "F32") {
        // Direct copy
        const float* src = reinterpret_cast<const float*>(
            cpu_data_.data() + meta.data_start);
        cudaMemcpy(d_ptr, src, n * sizeof(float), cudaMemcpyHostToDevice);
    } else if (meta.dtype == "F16") {
        // Convert FP16 → FP32 on CPU before uploading
        const uint16_t* src = reinterpret_cast<const uint16_t*>(
            cpu_data_.data() + meta.data_start);
        vector<float> buf(n);
        for (size_t i = 0; i < n; i++) {
            // IEEE 754 FP16 → FP32 conversion
            uint16_t h = src[i];
            uint32_t sign     = (h >> 15) & 1;
            uint32_t exponent = (h >> 10) & 0x1F;
            uint32_t mantissa =  h        & 0x3FF;
            uint32_t f;
            if (exponent == 0) {
                f = (sign << 31) | (mantissa << 13);
            } else if (exponent == 0x1F) {
                f = (sign << 31) | (0xFF << 23) | (mantissa << 13);
            } else {
                f = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
            }
            memcpy(&buf[i], &f, sizeof(float));
        }
        cudaMemcpy(d_ptr, buf.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        throw runtime_error("Unsupported dtype: " + meta.dtype);
    }

    gpu_ptrs_.push_back(d_ptr);  // track for cleanup
    return d_ptr;
}

vector<size_t> WeightLoader::shape(const string& name) const {
    auto it = metas_.find(name);
    if (it == metas_.end())
        throw runtime_error("Tensor not found: " + name);
    return it->second.shape;
}

bool WeightLoader::has(const string& name) const {
    return metas_.count(name) > 0;
}

void WeightLoader::print_tensors() const {
    for (auto& [name, meta] : metas_) {
        cout << "  " << name << " | " << meta.dtype << " | [";
        for (size_t i = 0; i < meta.shape.size(); i++) {
            cout << meta.shape[i];
            if (i + 1 < meta.shape.size()) cout << ", ";
        }
        cout << "]\n";
    }
}

WeightLoader::~WeightLoader() {
    for (float* ptr : gpu_ptrs_) {
        cudaFree(ptr);
    }
}