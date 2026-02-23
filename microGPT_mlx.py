"""
The most atomic way to train and inference a GPT on Apple Metal.
A 1:1 port of karpathy/microgpt.py — only change: Value scalars → MLX arrays.
"""

import os, math, random
import mlx.core as mx
import mlx.nn as nn

# Hyperparameters
n_embd, n_layer, n_head, block_size = 16, 1, 4, 8
head_dim = n_embd // n_head

# Dataset
if not os.path.exists('input.txt'):
    import urllib.request
    urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt', 'input.txt')
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)

# Tokenizer
chars = ['<BOS>', '<EOS>'] + sorted(set(''.join(docs)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
BOS, EOS = stoi['<BOS>'], stoi['<EOS>']
print(f"vocab size: {vocab_size}, num docs: {len(docs)}")

# Parameters — same names as original, now MLX arrays instead of [[Value]]
def matrix(nout, nin, std=0.02):
    return mx.random.normal((nout, nin)) * std

state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd, std=0)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd, std=0)
params = list(state_dict.values())
print(f"num params: {sum(p.size for p in params)}")

# Model — same structure as original, loops replaced by matmul
def linear(x, w):        return x @ w.T
def softmax(x):          return mx.softmax(x, axis=-1)
def rmsnorm(x):          return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)

def gpt(token_id, pos_id, keys, values):
    x = state_dict['wte'][token_id] + state_dict['wpe'][pos_id % block_size]  # (n_embd,)

    for li in range(n_layer):
        # 1) Multi-head attention
        xr = x
        x  = rmsnorm(x)
        q  = linear(x, state_dict[f'layer{li}.attn_wq'])
        k  = linear(x, state_dict[f'layer{li}.attn_wk'])
        v  = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k); values[li].append(v)
        K  = mx.stack(keys[li])    # (t, n_embd)
        V  = mx.stack(values[li])  # (t, n_embd)
        # split heads and attend
        q  = q.reshape(n_head, head_dim)
        K  = K.reshape(-1, n_head, head_dim).transpose(1, 0, 2)  # (n_head, t, head_dim)
        V  = V.reshape(-1, n_head, head_dim).transpose(1, 0, 2)
        w  = softmax((q[:, None, :] @ K.transpose(0, 2, 1)).squeeze(1) / head_dim**0.5)  # (n_head, t)
        x  = (w[:, None, :] @ V).squeeze(1).reshape(n_embd)      # (n_embd,)
        x  = linear(x, state_dict[f'layer{li}.attn_wo']) + xr
        # 2) MLP
        xr = x
        x  = rmsnorm(x)
        x  = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x  = mx.maximum(x, 0) ** 2                                # ReLU²
        x  = linear(x, state_dict[f'layer{li}.mlp_fc2']) + xr

    return linear(x, state_dict['wte'])  # logits (vocab_size,)

# Loss
def loss_fn(sd, tokens):
    global state_dict
    state_dict = sd
    keys   = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(len(tokens) - 1):
        logits = gpt(tokens[pos_id], pos_id, keys, values)
        log_p  = logits - mx.logsumexp(logits)
        losses.append(-log_p[tokens[pos_id + 1]])
    return mx.mean(mx.stack(losses))

# Adam optimizer
lr, beta1, beta2, eps = 1e-2, 0.9, 0.95, 1e-8
m = {k: mx.zeros_like(v) for k, v in state_dict.items()}
v = {k: mx.zeros_like(v) for k, v in state_dict.items()}

grad_fn = mx.grad(loss_fn)

# Training loop
for step in range(1000):
    doc    = docs[step % len(docs)]
    tokens = ([BOS] + [stoi[ch] for ch in doc] + [EOS])[:block_size]
    if len(tokens) < 2: continue

    grads  = grad_fn(state_dict, tokens)
    lr_t   = lr * (1 - step / 1000)
    for k in state_dict:
        m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
        v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
        mh   = m[k] / (1 - beta1 ** (step + 1))
        vh   = v[k] / (1 - beta2 ** (step + 1))
        state_dict[k] -= lr_t * mh / (mx.sqrt(vh) + eps)
    mx.eval(state_dict)

    if (step + 1) % 100 == 0:
        loss = loss_fn(state_dict, tokens)
        mx.eval(loss)
        print(f"step {step+1} / 1000 | loss {loss.item():.4f}")

# Inference
print("\n--- generation ---")
for i in range(5):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id, out = BOS, []
    for pos_id in range(block_size):
        logits   = gpt(token_id, pos_id, keys, values)
        mx.eval(logits)
        token_id = random.choices(range(vocab_size), weights=mx.softmax(logits).tolist())[0]
        if token_id == EOS: break
        out.append(itos[token_id])
    print(f"sample {i}: {''.join(out)}")
