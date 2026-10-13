import torch
from model import BigramLanguageModel

batch_size    = 32
block_size    = 8
max_iters     = 3000
eval_interval = 300
learning_rate = 1e-2
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters    = 200

torch.manual_seed(1337)

# input.txt file bata text padney ra character-level vocabulary banaune
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars      = sorted(list(set(text)))
vocab_size = len(chars)

stoi   = {ch: i for i, ch in enumerate(chars)}
itos   = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data       = torch.tensor(encode(text), dtype=torch.long)
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]


def get_batch(split: str):
    src = train_data if split == 'train' else val_data
    ix  = torch.randint(len(src) - block_size, (batch_size,))
    x   = torch.stack([src[i:i + block_size]         for i in ix])
    y   = torch.stack([src[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y      = get_batch(split)
            _, loss   = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model     = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# model lai train garney loop - har step ma ek batch liyera loss calculate garney
for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step:>4d}: train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    xb, yb  = get_batch('train')
    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context   = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated))