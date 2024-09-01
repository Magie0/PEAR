# PEAR: Parallel and Efficient Accelerated compressoR

PEAR (Parallel and Efficient Accelerated compressoR) is a novel learning-based compression framework designed to improve data storage efficiency by addressing the challenges of high computational costs and memory usage in existing compressors. This repository contains the implementation of PEAR, along with experimental scripts and datasets for reproducing the results.

The code of performer is from https://github.com/mynotwo/Faster-and-Stronger-Lossless-Compression-with-Optimized-Autoregressive-Framework

## Overview

Advances in general-purpose lossless compression have significantly enhanced data storage efficiency, particularly with the exponential growth in data generation and storage. However, existing learning-based compressors often struggle with high computational costs due to memory-intensive models and inefficient serial entropy coding. PEAR addresses these challenges with:

- A **lightweight probability estimation module** featuring:
  - **Byte-Level Fusion Block** for enhanced byte-level feature extraction.
  - **Batch-Adaptive Fusion Block** to optimize weight usage and reduce memory consumption.
- A **parallel encoding module** to accelerate entropy coding and improve overall compression speed.

## Features

- **Improved Compression Ratios**: PEAR achieves fine-grained data modeling, resulting in an average 3.4% improvement in compression ratio over state-of-the-art compressors.
- **Faster Compression Speed**: The parallel encoding module accelerates the compression process, providing a 130% increase in speed.
- **Reduced Memory Consumption**: Optimized for memory efficiency, PEAR reduces memory usage by 35% compared to existing methods.

## Dataset

To run PEAR, you will need to download the datasets used in our experiments. The datasets can be downloaded from the following link:

- [Dataset Download Link](<insert-your-link-here>)

Please ensure you have the datasets downloaded and properly placed in your working directory before running the compression and decompression processes.

## Usage
```bash
git clone https://github.com/Magie0/PEAR.git
cd PEAR

PEAR integrates both compression and decompression in a single command, simplifying the workflow. To run the process, use the following command:

```bash
python PEARencoding.py --input_dir <input_directory> --prefix <prefix> --gpu_id <gpu_id> --batch_size <batch_size>
