import torch
from model import GPT2, GPTConfig
import tiktoken

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    return device

class DataLoaderLite:
    def __init__(self, batch_size, block_size, split="train"):
        self.B = batch_size
        self.T = block_size
        self.split = split

        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        tokenizer = tiktoken.get_encoding("gpt2")
        self.tokens = tokenizer.encode(text)
        self.current_pos = 0

    def next_batch(self):
        B, T = self.B, self.T
        # get the next batch of tokens
        x = torch.tensor(self.tokens[self.current_pos:self.current_pos+B*T], dtype=torch.long).view(B, T)
        y = torch.tensor(self.tokens[self.current_pos+1:self.current_pos+B*T+1], dtype=torch.long).view(B, T)
        self.current_pos += B*T

        # if we've reached the end of the dataset, reset the position
        if self.current_pos + B*T > len(self.tokens):
            self.current_pos = 0

        return (x,y)
        
def train():
    device = get_device()
    config = GPTConfig()
    model = GPT2(config)
    model.to(device)

    max_steps = 100

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_loader = DataLoaderLite(batch_size=4, block_size=32)

    # a simple train loop
    for i in range(max_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"step {i}: loss {loss.item(): .4f}")


if __name__ == '__main__':
    train()