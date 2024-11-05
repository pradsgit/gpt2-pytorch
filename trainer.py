import time
import torch
from model import GPT2, GPTConfig
import tiktoken
import torch._dynamo

torch._dynamo.config.suppress_errors = True

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    return device

# random seed setting
torch.manual_seed(1442)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1442)

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
        self.current_pos += B * T

        # if we've reached the end of the dataset, reset the position
        if self.current_pos + B*T > len(self.tokens):
            self.current_pos = 0

        return (x,y)
        
def train():
    device = get_device()
    config = GPTConfig()
    model = GPT2(config)
    model.to(device)
    model = torch.compile(model)

    torch.set_float32_matmul_precision('high') # using mixed-precision(TF32 or FP16). works only for NVIDIA GPUs cause of specialized hardware units(Tensor Cores) in their gpus. for faster computation on GPUs

    max_steps = 50
    B, T = 2, 128
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    train_loader = DataLoaderLite(batch_size=B, block_size=T)

    # a simple train loop
    for i in range(max_steps):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if torch.cuda.is_available():
            print('cuda available. using autocast to bf16')
            with torch.autocast(device_type=device, dtype=torch.bfloat16): # using mixed-precision(BF16). works for NVIDIA GPUs starting from Ampere architecture and for Apple M1/M2 GPUs. for faster computation on GPUs
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if device == 'mps':
            torch.mps.synchronize() # wait for all the computation to be done MPS in that iteration
        else:
            torch.cuda.synchronize() # wait for all the computation to be done GPUs in that iteration
        t1 = time.time()
        dt = (t1 - t0) * 1000 # in ms
        tokens_per_sec = train_loader.B * train_loader.T / (t1 - t0)
        print(f"step {i}: loss {loss.item(): .4f} time elapsed: {dt:.2f}ms tokens/sec: {tokens_per_sec: .2f}")


if __name__ == '__main__':
    train()