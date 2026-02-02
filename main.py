import src.bytetok as btok
from datasets import load_dataset


def main() -> None:
    """Load and preprocess sci-fi books dataset for tokenizer training."""
    # preprocessing
    ds = load_dataset("stevez80/Sci-Fi-Books-gutenberg", split="train")
    lines = ds[:]["text"]
    print(f"number of lines {len(lines)}")
    text = "".join(lines)
    print(f"number of chars {len(text)}")
    vocab_size = 10_000

    # train tokenizer on dataset
    tok = btok.get_tokenizer("gpt4")
    tok.train(text, vocab_size, verbose=True)


if __name__ == "__main__":
    main()
