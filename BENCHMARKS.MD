## OLD Benchmarks

Benchmarks were run on Linux x86_64 with an Intel Core i7-12700H (20 cores @ 4.70 GHz) and 32GB DDR5 RAM.

Dataset: [Sci-Fi Books (Gutenberg)](https://huggingface.co/datasets/stevez80/Sci-Fi-Books-gutenberg).

| Corpus Size | Vocab Size | Training Time | Encoding Throughput | Decoding Throughput | Compression Ratio | Size Reduction |
| ----------- | ---------- | ------------- | ------------------- | ------------------- | ----------------- | -------------- |
| 88.85 MB    | 25,000     | 3.3 mins      | 2.85 MB/sec         | 19.2M tokens/sec    | 1.43x             | 30.3%          |
| 216.96 MB   | 10,000     | 8.7 mins      | 2.73 MB/sec         | 17.1M tokens/sec    | 1.60x             | 37.7%          |
| 216.96 MB   | 25,000     | 9.65 mins     | 2.72 MB/sec         | 17.0M tokens/sec    | 1.68x             | 40.6%          |
| 216.96 MB   | 50,000     | 10.7 mins     | 2.63 MB/sec         | 16.4M tokens/sec    | 1.75x             | 42.7%          |
| 326.96 MB   | 50,000     | 17.5 mins     | 2.69 MB/sec         | 7.02M tokens/sec    | 1.44x             | 30.7%          |

## NEW Benchmarks (after adding parallel encoding)

encoding used with 20 workers and auto mode.

| Corpus Size | Vocab Size | Training Time | Encoding Throughput | Decoding Throughput | Compression Ratio | Size Reduction |
| ----------- | ---------- | ------------- | ------------------- | ------------------- | ----------------- | -------------- |
| 216.96 MB   | 10,000     | 18.9 secs     | 9.36 MB/sec         | 53.7M tokens/sec    | 1.32x             | 24.3%          |
| 216.96 MB   | 25,000     | 15.95 mins    | 9.45 MB/sec         | 45.2M tokens/sec    | 1.68x             | 40.6%          |
| 216.96 MB   | 50,000     | 23.1 secs     | 9.25 MB/sec         | 54.6M tokens/sec    | 1.38x             | 27.6%          |
| 326.96 MB   | 50,000     | 23.6 secs     | 10.37 MB/sec        | 52.8M tokens/sec    | 1.39x             | 27.8%          |
| 132.36 MB   | 10,000     | 19.6 secs     | 10.41 MB/sec        | 54.3M tokens/sec    | 1.34x             | 25.1%          |
