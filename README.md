# PEAR: Parallel and Efficient Accelerated compressoR

PEAR (Parallel and Efficient Accelerated compressoR) is a novel learning-based compression framework designed to improve data storage efficiency by addressing the challenges of high computational costs and memory usage in existing compressors. This repository contains the implementation of PEAR, along with experimental scripts and datasets for reproducing the results.

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

## Installation

To install and run PEAR, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/pear-compressor.git
    cd pear-compressor
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the compression script**:
    ```bash
    python run_compression.py --input <input_file> --output <output_file>
    ```

## Usage

### Compression

To compress a file using PEAR:

```bash
python run_compression.py --input <input_file> --output <output_file>
