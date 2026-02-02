use bpe::PyBPETrainer;

fn main() {
    // Example: "abaabbaa"
    // a=0, b=1
    let tokens = vec![0, 1, 0, 0, 1, 1, 0, 0];

    let mut trainer = PyBPETrainer::new(tokens, 2);

    println!("Initial:");
    trainer.print_state();

    trainer.train(3);

    println!("\nFinal result:");
    trainer.print_state();

    println!("\nFinal tokens: {:?}", trainer.get_tokens());
}
