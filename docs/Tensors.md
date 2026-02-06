# Tensor

> **NOTE:** Slices and views are not diff or copy of tensors but the actual tensor indicated with the offset 

Shape is the dimension of a tensor
Stride 

## View

view handles shaping and reshaping of rows and colums

```cpp
// Suppose we have a 2x3 tensor
Tensor t({2, 3});  
// Memory layout (flattened): [a, b, c, d, e, f]

// Make a view that treats it as 3x2
Tensor v = t.view({3, 2});

// v sees memory like:
// [[a, b],
//  [c, d],
//  [e, f]]
```

## View Slicing Example

```cpp
Tensor t({4}); // [0, 1, 2, 3]
Tensor s = t.view({2, 2}); // reshape into 2x2
Tensor slice = s.view({2}); // pick first row
```

## Stride Calculation

### **What this code does**

```cpp
stride_.resize(shape.size()); // one stride per dimension
size_t s = 1;
for (int i = shape.size() - 1; i >= 0; --i) {
    stride_[i] = s;
    s *= shape[i];
}
```

* `shape.size()` = number of axes (dimensions) of the tensor.
* `stride_[i]` = **how many scalars in storage you need to skip to move 1 step along axis i**.
* `s` accumulates the **total size of the dimensions after i**, to calculate stride for the current dimension.

---

### **Example: 32 RGB images, 100×100 pixels**

* `shape = [32, 3, 100, 100]`
* Loop runs **from last dimension to first** (width → height → channel → batch)

Step by step:

1. `i = 3` (width, 100) → `stride[3] = s = 1` → move 1 scalar to go to the next column
2. `s *= shape[3] = 1 * 100 = 100`
3. `i = 2` (height, 100) → `stride[2] = s = 100` → move 100 scalars to go to the next row
4. `s *= shape[2] = 100 * 100 = 10,000`
5. `i = 1` (channels, 3) → `stride[1] = s = 10,000` → move 10,000 scalars to go to next channel
6. `s *= shape[1] = 10,000 * 3 = 30,000`
7. `i = 0` (batch, 32) → `stride[0] = s = 30,000` → move 30,000 scalars to go to next image

✅ **Resulting strides:** `[30,000, 10,000, 100, 1]`

---

### **How it connects to tensors / chairs / images**

| Concept           | Memory / Chair analogy                                                            |
| ----------------- | --------------------------------------------------------------------------------- |
| Scalar            | one chair = one pixel                                                             |
| Vector            | row of chairs = row of pixels, stride = 1                                         |
| Matrix            | floor of chairs = 100×100 pixels, stride along row = 100, stride along column = 1 |
| 3D Tensor (RGB)   | stack 3 floors = 3 channels, stride along channel = 10,000                        |
| 4D Tensor (Batch) | stack 32 images = batch, stride along batch = 30,000                              |

* **Stride tells the tensor:** “To move along this axis, how many scalars do I skip in the flat storage?”
* **Offset tells the tensor:** “Where do I start reading in the flat storage?”
