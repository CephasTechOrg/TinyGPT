from pathlib import Path

DATA = [
    ("What is a stack?",
     "A stack is a linear data structure that follows Last In First Out order."),
    ("What is a queue?",
     "A queue is a linear data structure that follows First In First Out order."),
    ("What is Big O notation?",
     "Big O notation describes the time or space complexity of an algorithm."),
    ("What is a linked list?",
     "A linked list is a data structure made of nodes where each node points to the next."),
    ("What does CPU mean?",
     "CPU stands for Central Processing Unit."),
    ("What is an algorithm?",
     "An algorithm is a step by step procedure to solve a problem."),
]

def main():
    out_dir = Path("data/toy")
    out_dir.mkdir(parents=True, exist_ok=True)

    text = ""
    for q, a in DATA:
        text += f"User: {q}\nAssistant: {a}\n\n"

    (out_dir / "train.txt").write_text(text)
    print("Toy dataset written to data/toy/train.txt")

if __name__ == "__main__":
    main()
