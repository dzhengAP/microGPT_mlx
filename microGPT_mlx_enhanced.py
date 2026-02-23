"""
microGPT_mlx.py
The most atomic way to train and inference a GPT on Apple Metal.
A 1:1 port of karpathy/microgpt.py — only change: Value scalars → MLX arrays.

Improvements over v1 (not enhanced):
  - Larger model (n_embd=32, n_layer=2) for better generation quality
  - Mini-batch training (batch_size=8) for more stable loss
  - Cosine LR schedule with warmup
  - Gradient clipping
  - Periodic generation during training to monitor quality
  - Explicit mx.eval on grads for cleaner debugging
  - Top-k sampling in inference
"""

import os, math, random
import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
n_embd     = 32
n_layer    = 2
n_head     = 4
block_size = 16
head_dim   = n_embd // n_head

batch_size = 8
max_steps  = 5000

lr_max     = 3e-3
lr_min     = 1e-4
warmup     = 200

beta1, beta2, eps = 0.9, 0.95, 1e-8
grad_clip  = 1.0

top_k      = 5   # sampling top-k during inference

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
if not os.path.exists('input.txt'):
    import urllib.request
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt',
        'input.txt'
    )
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
chars     = ['<BOS>', '<EOS>'] + sorted(set(''.join(docs)))
vocab_size = len(chars)
stoi      = {ch: i for i, ch in enumerate(chars)}
itos      = {i: ch for i, ch in enumerate(chars)}
BOS, EOS  = stoi['<BOS>'], stoi['<EOS>']
print(f"vocab size: {vocab_size}, num docs: {len(docs)}")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
def matrix(nout, nin, std=0.02):
    return mx.random.normal((nout, nin)) * std

state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd, std=0)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd, std=0)

params = list(state_dict.values())
print(f"num params: {sum(p.size for p in params)}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def linear(x, w):   return x @ w.T
def softmax(x):      return mx.softmax(x, axis=-1)
def rmsnorm(x):      return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)

def gpt(token_id, pos_id, keys, values):
    """Single-token forward pass with KV cache (autoregressive, accidentally causal)."""
    x = state_dict['wte'][token_id] + state_dict['wpe'][pos_id % block_size]  # (n_embd,)

    for li in range(n_layer):
        # --- Multi-head self-attention ---
        xr = x
        x  = rmsnorm(x)
        q  = linear(x, state_dict[f'layer{li}.attn_wq'])
        k  = linear(x, state_dict[f'layer{li}.attn_wk'])
        v  = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        K  = mx.stack(keys[li])                                    # (t, n_embd)
        V  = mx.stack(values[li])                                  # (t, n_embd)

        # split heads
        q  = q.reshape(n_head, head_dim)                           # (n_head, head_dim)
        K  = K.reshape(-1, n_head, head_dim).transpose(1, 0, 2)   # (n_head, t, head_dim)
        V  = V.reshape(-1, n_head, head_dim).transpose(1, 0, 2)   # (n_head, t, head_dim)

        # scaled dot-product attention
        w  = softmax(
            (q[:, None, :] @ K.transpose(0, 2, 1)).squeeze(1) / head_dim**0.5
        )                                                           # (n_head, t)
        x  = (w[:, None, :] @ V).squeeze(1).reshape(n_embd)       # (n_embd,)
        x  = linear(x, state_dict[f'layer{li}.attn_wo']) + xr

        # --- MLP with ReLU² ---
        xr = x
        x  = rmsnorm(x)
        x  = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x  = mx.maximum(x, 0) ** 2                                 # ReLU²
        x  = linear(x, state_dict[f'layer{li}.mlp_fc2']) + xr

    return linear(x, state_dict['wte'])                            # logits (vocab_size,)

# ---------------------------------------------------------------------------
# Loss (mini-batch)
# ---------------------------------------------------------------------------
def loss_fn(sd, batch):
    """
    batch: list of token-id lists (variable length, each >= 2)
    Returns scalar mean cross-entropy loss.
    """
    global state_dict
    state_dict = sd
    doc_losses = []
    for tokens in batch:
        keys   = [[] for _ in range(n_layer)]
        values = [[] for _ in range(n_layer)]
        losses = []
        for pos_id in range(len(tokens) - 1):
            logits = gpt(tokens[pos_id], pos_id, keys, values)
            log_p  = logits - mx.logsumexp(logits)
            losses.append(-log_p[tokens[pos_id + 1]])
        doc_losses.append(mx.mean(mx.stack(losses)))
    return mx.mean(mx.stack(doc_losses))

# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------
def get_lr(step):
    if step < warmup:
        return lr_max * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

# ---------------------------------------------------------------------------
# Adam optimizer state
# ---------------------------------------------------------------------------
m_state = {k: mx.zeros_like(v) for k, v in state_dict.items()}
v_state = {k: mx.zeros_like(v) for k, v in state_dict.items()}

grad_fn = mx.grad(loss_fn)

# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def generate(top_k=top_k):
    keys_g   = [[] for _ in range(n_layer)]
    values_g = [[] for _ in range(n_layer)]
    token_id, out = BOS, []
    for pos_id in range(block_size):
        logits   = gpt(token_id, pos_id, keys_g, values_g)
        mx.eval(logits)
        # top-k sampling
        logits_np = logits.tolist()
        top_pairs = sorted(enumerate(logits_np), key=lambda x: x[1], reverse=True)[:top_k]
        ids, vals = zip(*top_pairs)
        probs     = mx.softmax(mx.array(list(vals))).tolist()
        token_id  = random.choices(list(ids), weights=probs)[0]
        if token_id == EOS:
            break
        out.append(itos[token_id])
    return ''.join(out)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print(f"\nTraining for {max_steps} steps, batch_size={batch_size} ...\n")

for step in range(max_steps):
    # sample a mini-batch of docs
    batch_docs = random.choices(docs, k=batch_size)
    batch = []
    for doc in batch_docs:
        tokens = ([BOS] + [stoi[ch] for ch in doc] + [EOS])[:block_size]
        if len(tokens) >= 2:
            batch.append(tokens)
    if not batch:
        continue

    grads = grad_fn(state_dict, batch)

    # gradient clipping
    grad_norm = mx.sqrt(
        mx.sum(mx.stack([mx.sum(g * g) for g in grads.values()]))
    )
    mx.eval(grad_norm)
    scale = min(1.0, grad_clip / (grad_norm.item() + 1e-6))

    lr_t  = get_lr(step)
    t     = step + 1
    for k in state_dict:
        g          = grads[k] * scale
        m_state[k] = beta1 * m_state[k] + (1 - beta1) * g
        v_state[k] = beta2 * v_state[k] + (1 - beta2) * g ** 2
        mh         = m_state[k] / (1 - beta1 ** t)
        vh         = v_state[k] / (1 - beta2 ** t)
        state_dict[k] = state_dict[k] - lr_t * mh / (mx.sqrt(vh) + eps)

    mx.eval(state_dict)

    if (step + 1) % 500 == 0:
        loss = loss_fn(state_dict, batch)
        mx.eval(loss)
        samples = ', '.join(generate() for _ in range(3))
        print(f"step {step+1:5d}/{max_steps} | lr {lr_t:.2e} | loss {loss.item():.4f} | {samples}")

# ---------------------------------------------------------------------------
# Final generation
# ---------------------------------------------------------------------------
print("\n--- final generation ---")
for i in range(10):
    print(f"  sample {i:2d}: {generate()}")
