import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters - model lai train garna lagney settings haru
batch_size = 32 # ek patak ma kati wota sequence lai sanga-sangai process garney?
block_size = 8 # prediction garna ko lagi ab maximum kati wota character herna sakcha?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------ settings yaha samma ------------

torch.manual_seed(1337)

# yo link bata input.txt file download garney (tinyshakespeare dataset)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# yo text ma bhetiyeko sabai farak-farak (unique) characters haru
chars = sorted(list(set(text)))
vocab_size = len(chars)
# character lai number ma ra number lai character ma badalne mapping banaune
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: string lincha, number ko list dincha
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: number ko list lincha, string dincha

# Data lai train ra test ma vibhajan (split) garney
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # pahilo 90% train ko lagi, baki 10% validation ko lagi
train_data = data[:n]
val_data = data[n:]

# data load garne function
def get_batch(split):
    # sano batch ko data banaune - input x ra target y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ekdam sajilo bigram bhasha model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # haryek token le arko token ko lagi logits sidhai lookup table bata padcha
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx ra targets duita-i (B,T) aakar ko integer tensor hun
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx bhaney ko ahile ko context ma bhayeko indices ko (B, T) array ho
        for _ in range(max_new_tokens):
            # model bata prediction nikalne
            logits, loss = self(idx)
            # last time step ma matra dhyan dine
            logits = logits[:, -1, :] # becomes (B, C)
            # softmax lagayera probability nikalne
            probs = F.softmax(logits, dim=-1) # (B, C)
            # probability distribution bata ek wota sample channe
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # chaneko sample lai sequence ma jodne
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# PyTorch optimizer banaune (model lai sikaune kaam garcha)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # bich-bich ma train ra val set ma loss check garne
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # ek batch data sample garne
    xb, yb = get_batch('train')

    # loss calculate garne ra model lai update garne
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# train bhayeko model bata naya text generate garne
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))