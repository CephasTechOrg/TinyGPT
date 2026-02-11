from pathlib import Path

DATA = [
    # Data Structures
    ("What is a stack?",
     "A stack is a linear data structure that follows Last In First Out order."),

    ("What is a queue?",
     "A queue is a linear data structure that follows First In First Out order."),

    ("What is a linked list?",
     "A linked list is a data structure made of nodes where each node points to the next node."),

    ("What is an array?",
     "An array is a data structure that stores elements in contiguous memory locations."),

    ("What is a tree?",
     "A tree is a hierarchical data structure with a root node and child nodes."),

    ("What is a graph?",
     "A graph is a data structure made of nodes connected by edges."),

    # Algorithms
    ("What is an algorithm?",
     "An algorithm is a step by step procedure to solve a problem."),

    ("What is sorting?",
     "Sorting is the process of arranging data in a specific order."),

    ("What is searching?",
     "Searching is the process of finding an element in a data structure."),

    ("What is binary search?",
     "Binary search is an efficient algorithm that finds an element in a sorted array."),

    # Complexity
    ("What is Big O notation?",
     "Big O notation describes the time or space complexity of an algorithm."),

    ("What does O(n) mean?",
     "O(n) means the algorithm runs in linear time."),

    ("What does O(1) mean?",
     "O(1) means the algorithm runs in constant time."),

    ("What does O(n squared) mean?",
     "O(n squared) means the algorithm runs in quadratic time."),

    # Computer Basics
    ("What is a CPU?",
     "CPU stands for Central Processing Unit and it executes program instructions."),

    ("What is RAM?",
     "RAM is temporary memory used to store data for running programs."),

    ("What is an operating system?",
     "An operating system manages hardware and software resources on a computer."),

    ("What is a process?",
     "A process is a program that is currently being executed."),

    ("What is a thread?",
     "A thread is the smallest unit of execution within a process."),

    # Programming
    ("What is a variable?",
     "A variable is a named storage location used to store data."),

    ("What is a function?",
     "A function is a reusable block of code that performs a specific task."),

    ("What is a loop?",
     "A loop is used to repeat a block of code multiple times."),

    ("What is a conditional statement?",
     "A conditional statement executes code based on a condition."),

    ("What is debugging?",
     "Debugging is the process of finding and fixing errors in code."),

    # AI / ML Basics
    ("What is artificial intelligence?",
     "Artificial intelligence is the simulation of human intelligence by machines."),

    ("What is machine learning?",
     "Machine learning is a subset of AI where machines learn from data."),

    ("What is a neural network?",
     "A neural network is a model inspired by the human brain made of connected layers."),

    ("What is training in machine learning?",
     "Training is the process of adjusting model parameters using data."),

    ("What is a model in machine learning?",
     "A model is a mathematical function that maps inputs to outputs."),

    ("What is overfitting?",
     "Overfitting happens when a model memorizes data instead of learning patterns."),

    ("What is a dataset?",
     "A dataset is a collection of data used for training or testing models."),

    # Chat behavior (VERY IMPORTANT)
    ("Hello",
     "Hello! How can I help you today?"),

    ("Hi",
     "Hi! What would you like to learn today?"),

    ("What can you do?",
     "I can answer questions and explain basic computer science and AI concepts."),

    ("Who are you?",
     "I am a small AI chatbot built using a transformer model."),
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
