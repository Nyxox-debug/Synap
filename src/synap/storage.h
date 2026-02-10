#pragma once
#include <cstddef>
#include <memory>

struct Storage {
    float* data;
    size_t size;

    Storage(size_t size)
        : size(size) {
        data = new float[size]();
    }

    ~Storage() {
        delete[] data;
    }
};

using StoragePtr = std::shared_ptr<Storage>;
