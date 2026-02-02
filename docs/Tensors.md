# Tensor

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
