import torch
from pathlib import Path
from src.data.tokenizer import CharTokenizer
from src.model.gpt import TinyGPT

@torch.no_grad()
def generate(model, tokenizer, idx, max_new_tokens=200, temperature=0.8):
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.pos_emb.num_embeddings:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

        # stop if it begins a new turn
        text = tokenizer.decode(idx[0].tolist())
        if "\nUser:" in text:
            break

    return idx

def main():
    text = Path("data/toy/train.txt").read_text()
    tokenizer = CharTokenizer(text)

    ckpt = torch.load("checkpoints/latest.pt", map_location="cpu")
    config = ckpt["config"]

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        context_length=config["context_length"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
    )
    model.load_state_dict(ckpt["model_state"])

    print("\nTinyChatGPT ready. Type 'exit' to quit.\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        prompt = f"User: {user_input}\nAssistant:"
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

        out = generate(model, tokenizer, idx, max_new_tokens=200, temperature=0.2)
        reply = tokenizer.decode(out[0].tolist())
        print("\nAssistant: " + reply.split("Assistant:")[-1].strip() + "\n")

if __name__ == "__main__":
    main()
