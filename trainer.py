import torch
from model import GPT2, GPTConfig

def train():
    config = GPTConfig()
    model = GPT2(config)
    # a test model run
    B, T = 4, 32
    x = torch.randint(0, config.vocab_size, (B, T))
    print(f"x shape is {x.shape}")
    logits = model(x)
    print(logits.shape)

if __name__ == '__main__':
    train()