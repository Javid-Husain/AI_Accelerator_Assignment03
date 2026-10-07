# AI Accelerator Design – Assignment 3
## BCHW Tensor Flattening and Reconstruction

**Name:** Md Javid  
**Roll No:** CS25M210

---

## 1. Objective

This assignment implements and verifies a manual method for **flattening a 4-D BCHW tensor** into a 2-D representation and reconstructing the original tensor from that flattened representation.

The tensor follows the layout:

```text
B × C × H × W
```

where:

- **B** = Batch size
- **C** = Number of channels
- **H** = Height
- **W** = Width

The implementation performs the address mapping explicitly using loops and integer arithmetic. It does **not** use `reshape()`, `view()`, `flatten()`, or `ravel()` for the actual flattening/reconstruction algorithm.

---

## 2. Address Mapping

The input tensor `I` has shape:

```text
B × C × H × W
```

Each batch element becomes one row of the flattened tensor `F`, whose shape is:

```text
B × (C × H × W)
```

For an element at position:

```text
(b, c, h, w)
```

the corresponding flattened index is:

```text
k = c × H × W + h × W + w
```

with:

```text
0 ≤ k < C × H × W
```

The inverse mapping reconstructs the original coordinates from `k`:

```text
c = k // (H × W)

h = (k % (H × W)) // W

w = k % W
```

The batch index `b` remains unchanged because it is the row index in both representations.

This mapping follows **row-major ordering**, where `W` varies fastest, followed by `H`, and then `C`.

---

## 3. Implementation Approach

The flattening procedure conceptually performs:

```text
F = empty(B, C × H × W)

for b in 0 ... B-1:
    for c in 0 ... C-1:
        for h in 0 ... H-1:
            for w in 0 ... W-1:
                k = c × H × W + h × W + w
                F[b, k] = I[b, c, h, w]
```

The reconstruction performs:

```text
R = empty(B, C, H, W)

for b in 0 ... B-1:
    for k in 0 ... C × H × W - 1:
        c = k // (H × W)
        h = (k % (H × W)) // W
        w = k % W
        R[b, c, h, w] = F[b, k]
```

Arrays are allocated and accessed element-by-element. The algorithm itself does not rely on high-level tensor reshaping operations.

---

## 4. Manual Indexing Example

For:

```text
B = 1
C = 2
H = 2
W = 3
```

we have:

```text
H × W = 6
C × H × W = 12
```

The flattened index becomes:

```text
k = 6c + 3h + w
```

Example mappings include:

| `(b,c,h,w)` | `k` | Value |
|---|---:|---:|
| `(0,0,0,0)` | 0 | 10 |
| `(0,0,0,2)` | 2 | 12 |
| `(0,0,1,0)` | 3 | 13 |
| `(0,0,1,2)` | 5 | 15 |
| `(0,1,0,0)` | 6 | 20 |
| `(0,1,0,1)` | 7 | 21 |
| `(0,1,1,2)` | 11 | 25 |

The resulting flattened row is:

```text
[10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25]
```

Here:

- Channel 0 occupies indices `0–5`
- Channel 1 occupies indices `6–11`
- Within each channel, the rows of width `W = 3` are stored consecutively

The reconstruction successfully returns the original tensor.

---

## 5. Experiments

The implementation was tested with:

- **B = 4** for all experiments
- MNIST test samples using their native dimensions
- CIFAR-10 test samples using their native dimensions
- Synthetic feature maps with:
  - `H = W = 16`
  - `float32` values drawn from `N(0,1)`
  - Channel count swept from `C = 8` to `C = 500`

The reconstruction error is evaluated using:

```text
E = I - R
```

Two metrics are reported:

- **Emax** – maximum absolute reconstruction error
- **MAE** – mean absolute error

### Results

| Dataset / Input | B | C | H | W | CHW | Emax | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| MNIST / test samples | 4 | 1 | 28 | 28 | 784 | 0 | 0 |
| CIFAR-10 / test samples | 4 | 3 | 32 | 32 | 3072 | 0 | 0 |
| Synthetic fmap1 | 4 | 8 | 16 | 16 | 2048 | 0 | 0 |
| Synthetic fmap2 | 4 | 16 | 16 | 16 | 4096 | 0 | 0 |
| Synthetic fmap3 | 4 | 32 | 16 | 16 | 8192 | 0 | 0 |
| Synthetic fmap4 | 4 | 64 | 16 | 16 | 16384 | 0 | 0 |
| Synthetic fmap5 | 4 | 128 | 16 | 16 | 32768 | 0 | 0 |
| Synthetic fmap6 | 4 | 256 | 16 | 16 | 65536 | 0 | 0 |
| Synthetic fmap7 | 4 | 500 | 16 | 16 | 128000 | 0 | 0 |

The reconstruction produced **zero maximum error and zero mean absolute error for every tested configuration**.

---

## 6. Independent Verification

As an independent verification step, the explicitly generated flattened output was compared against NumPy's:

```python
reshape(B, -1)
```

The two outputs matched exactly.

This comparison was used only as a verification mechanism and was not part of the manual flattening/reconstruction algorithm.

---

## 7. Correctness

The forward mapping is:

```text
k = c × H × W + h × W + w
```

and the reverse mapping is:

```text
c = k // (H × W)
h = (k % (H × W)) // W
w = k % W
```

These mappings are exact inverses of each other. Therefore, every `(c,h,w)` position maps to one unique `k`, and every valid `k` maps back to one unique `(c,h,w)` position.

Since the algorithm only copies values and performs no arithmetic on the tensor values themselves, no rounding or numerical transformation is introduced.

This explains why:

```text
Emax = 0
MAE  = 0
```

for all tested configurations.

---

## 8. Complexity

Every tensor element is visited once during flattening and once during reconstruction.

For a tensor containing:

```text
B × C × H × W
```

elements:

- **Flattening time:** `O(BCHW)`
- **Reconstruction time:** `O(BCHW)`
- **Combined time:** `O(BCHW)`

The flattened representation requires:

```text
O(BCHW)
```

storage, in addition to the original/reconstructed tensor.

The runtime grows approximately linearly with the total number of tensor elements.

---

## 9. Relevance to AI Accelerators

This address-generation scheme is directly relevant to accelerator data movement.

When a BCHW feature map is streamed into a linear buffer or supplied to hardware such as a systolic array, the address-generation logic needs to convert multi-dimensional tensor coordinates into sequential memory addresses.

The mapping:

```text
k = c × H × W + h × W + w
```

causes consecutive addresses to move through:

```text
W → H → C
```

That means:

1. Elements within a row are accessed consecutively.
2. Rows within a channel are accessed consecutively.
3. Channels are then laid out consecutively.

This type of address generation is analogous to the work performed by a **DMA engine or address-generation unit (AGU)** in an AI accelerator.

---

## 10. Key Takeaways

- Implemented manual BCHW → `B × CHW` flattening.
- Implemented the exact inverse reconstruction.
- Used explicit address calculation instead of high-level reshape operations.
- Verified the method on MNIST, CIFAR-10, and synthetic feature maps.
- Tested channel counts up to `C = 500`.
- Obtained `Emax = 0` and `MAE = 0` across all reported experiments.
- Independently verified the flattened output against NumPy.
- Demonstrated how the mapping can be used for accelerator-oriented memory addressing.

---

## Author

**Md Javid**  
**Roll No: CS25M210**

**AI Accelerator Design – Assignment 3**
