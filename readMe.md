# Deep learning linking mechanistic models to single-cell transcriptomics data reveals transcriptional bursting in response to DNA damage
![image](https://github.com/cogitoErgoSumH/DeepTX/blob/main/assets/deepTXlogo.png)
Code and data for paper  "Deep learning linking mechanistic models to single-cell transcriptomics data reveals transcriptional bursting in response to DNA damage".

## System Requirements

The Julia code used for training and inference of the model comes with versions.

    Julia Version 1.7.1
    Commit ac5cc99908 (2021-12-22 19:35 UTC)
    Platform Info:
    OS: Windows (x86_64-w64-mingw32)
    CPU: AMD Ryzen 5 4600H with Radeon Graphics
    WORD_SIZE: 64
    LIBM: libopenlibm
    LLVM: libLLVM-12.0.1 (ORCJIT, znver2)
## Installation
DeepTX can be installed through the Julia package manager:
```julia
using Pkg
Pkg.add("https://github.com/cogitoErgoSumH/DeepTX.git")
```

## Examples

```julia
using DeepTX
using CSV
using DataFrames

gene_exp_data = DataFrame(
    CSV.File(
        joinpath(DATA_DIR,"your_scRNA_seq_data.csv" ),
    ),
)
estimated_params = DeepTX.inference_parameters(gene_exp_data)
```
