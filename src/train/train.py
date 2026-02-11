import torch
import yaml
from pathlib import Path
from src.data.tokenizer import CharTokenizer
from src.model.gpt import TinyGPT

def load_data():
    text = Path("data/toy/train.txt").read_text()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    return data, tokenizer

def main():
    config = yaml.safe_load(open("configs/tiny_cpu.yaml"))
    torch.manual_seed(config["seed"])

    data, tokenizer = load_data()
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        context_length=config["context_length"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    model.train()
    for step in range(config["max_steps"]):
        idx = torch.randint(0, len(data) - config["context_length"] - 1,
                            (config["batch_size"],))
        x = torch.stack([data[i:i+config["context_length"]] for i in idx])
        y = torch.stack([data[i+1:i+config["context_length"]+1] for i in idx])

        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()

        if step % config["log_interval"] == 0:
            print(f"step {step} | loss {loss.item():.4f}")

    print("Training complete.")

if __name__ == "__main__":
    main()
