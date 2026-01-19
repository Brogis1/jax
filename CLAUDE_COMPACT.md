# JAX Agent Spec (Compact)

**Core**: Functional programming. Pure functions. Composable transformations. NO OOP patterns.

---

## IMPORTS
```python
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from functools import partial
```

---

## TYPE ANNOTATIONS (REQUIRED)

```python
# Inputs: ArrayLike | Outputs: Array
def f(x: ArrayLike, w: ArrayLike) -> Array:
    return x @ w

# Params pytree
Params = dict[str, Array]

# Shapes in comments
def attention(
    q: ArrayLike,  # (B, L, D)
    k: ArrayLike,  # (B, L, D)
) -> Array:       # (B, L, D)
    """..."""

# Callables
LossFn = Callable[[Params, ArrayLike, ArrayLike], Array]
```

---

## DOCSTRINGS (Google Style)

```python
def softmax(x: ArrayLike, axis: int = -1) -> Array:
    """Compute softmax along axis.

    Args:
        x: Input logits, shape (..., n).
        axis: Normalization axis. Default: -1.

    Returns:
        Probabilities summing to 1, shape (..., n).
    """
    x_max = jnp.max(x, axis=axis, keepdims=True)
    return jnp.exp(x - x_max) / jnp.sum(jnp.exp(x - x_max), axis=axis, keepdims=True)
```

**Rules**: Imperative verb. Include shapes. Example optional.

---

## PURE FUNCTIONS

**NEVER:**
- Global mutation
- `print()` (use `jax.debug.print`)
- Iterators
- Mutable defaults
- Side effects

```python
# WRONG
g = 0
def bad(x):
    global g; g += 1; print(x); return x

# RIGHT
def good(x: ArrayLike, counter: int) -> tuple[Array, int]:
    return jnp.asarray(x), counter + 1
```

---

## IMMUTABILITY

```python
# WRONG: x[0] = 1.0

# RIGHT
x = x.at[0].set(1.0)
x = x.at[1:3].add(5.0)
x = x.at[mask].multiply(2.0)

# NOTE: += rebinds, doesn't mutate (unlike NumPy)
```

---

## JIT

**Static shapes required.** No value-dependent control flow.

```python
# WRONG: @jax.jit def f(x): return x[x > 0]  # dynamic shape
# RIGHT: @jax.jit def f(x): return jnp.where(x > 0, x, 0)

# Static args for control flow
@partial(jax.jit, static_argnames=['n'])
def f(x, n: int):
    for i in range(n):  # OK: n static
        x = x + 1
    return x

# CRITICAL: Don't jit in loops!
# WRONG: for i in range(n): x = jax.jit(lambda: x+1)()  # recompiles!
# RIGHT: @jax.jit outside, reuse
```

---

## GRAD

```python
# Scalar output required
def loss(params, x, y):
    return jnp.mean((forward(params, x) - y) ** 2)

grads = jax.grad(loss)(params, x, y)  # w.r.t. arg 0
grads = jax.grad(loss, argnums=(0, 1))(params, x, y)  # multiple
loss, grads = jax.value_and_grad(loss)(params, x, y)  # both

# NaN gotcha with where: grads flow through BOTH branches!
# WRONG: jnp.where(x > 0, jnp.log(x), 0)  # log(0) computed!
# RIGHT: jnp.where(x > 0, jnp.log(jnp.where(x > 0, x, 1)), 0)
```

---

## VMAP

```python
# Single -> batch
def single(x, params): return params['w'] @ x
batched = jax.vmap(single, in_axes=(0, None))  # batch x, broadcast params

# Pairwise
pairwise = jax.vmap(jax.vmap(f, (None, 0)), (0, None))
```

---

## CONTROL FLOW

**Prefer `lax.scan`** (fully differentiable)

```python
def step(carry, x):
    new_carry = carry + x
    return new_carry, carry * x

final, outs = jax.lax.scan(step, init, xs)

# Conditional (both branches traced!)
jax.lax.cond(pred, true_fn, false_fn, operand)
jnp.where(cond, x, y)  # elementwise

# Loops
jax.lax.fori_loop(0, n, lambda i, v: v+i, init)
jax.lax.while_loop(cond_fn, body_fn, init)
```

| Construct | jit | grad(rev) |
|-----------|-----|-----------|
| Python if | static only | OK |
| lax.cond | OK | **OK** |
| lax.scan | OK | **OK** |
| lax.fori_loop | OK | fwd only* |
| lax.while_loop | OK | fwd only |

---

## RANDOM

```python
key = jax.random.key(42)
key, k1, k2 = jax.random.split(key, 3)  # ALWAYS split!
x = jax.random.normal(k1, (100,))
y = jax.random.uniform(k2, (100,))

# NEVER reuse keys: same key = same output
# Use random.fold_in(key, i) for deterministic per-iteration keys
```

---

## PYTREES

```python
params = {'l1': {'w': w1, 'b': b1}, 'l2': {'w': w2, 'b': b2}}
jax.tree.map(lambda p, g: p - lr*g, params, grads)
jax.tree.leaves(params)

# GOTCHA: tuples are nodes, not leaves!
# WRONG: jax.tree.map(jnp.ones, shapes)  # shapes=(2,3) calls ones(2), ones(3)
# RIGHT: convert to array first

# GOTCHA: None not a leaf
# jax.tree.leaves([None, None]) -> []
# Use is_leaf=lambda x: x is None
```

---

## STATE

```python
# WRONG: class with mutable state
# RIGHT: explicit state threading

def init(key): return {'w': jax.random.normal(key, (10,10))}
def forward(params, x): return params['w'] @ x
def update(params, grads, lr=0.01):
    return jax.tree.map(lambda p,g: p-lr*g, params, grads)

params = init(key)
for batch in data:
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    params = update(params, grads)
```

---

## CUSTOM DERIVATIVES

```python
# JVP (forward-mode, auto-transposes for reverse)
@jax.custom_jvp
def f(x):
    return jnp.log(1 + jnp.exp(x))

@f.defjvp
def f_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return f(x), (1 - 1/(1+jnp.exp(x))) * x_dot

# VJP (reverse-only)
@jax.custom_vjp
def g(x, lo, hi):
    return x  # identity forward

def g_fwd(x, lo, hi):
    return x, (lo, hi)

def g_bwd(res, g_bar):
    lo, hi = res
    return (jnp.clip(g_bar, lo, hi), None, None)

g.defvjp(g_fwd, g_bwd)

# GOTCHA: nondiff_argnums go FIRST in JVP/VJP sig, regardless of position!
```

---

## CRITICAL GOTCHAS

```python
# Out-of-bounds: NO ERROR, clamped!
jnp.arange(10)[11]  # -> 9

# Lists as args: slow, convert first
# WRONG: jnp.sum([1,2,3])
# RIGHT: jnp.sum(jnp.array([1,2,3]))

# Closure capture: frozen at trace
# WRONG: scale=2; @jax.jit def f(x): return x*scale; scale=3  # still uses 2!
# RIGHT: pass as arg

# Iterators: don't work
# WRONG: it=iter(range(10)); lax.fori_loop(..., next(it), ...)
# RIGHT: use arrays

# Float64: disabled by default
jax.config.update("jax_enable_x64", True)

# Subnormals: flushed to zero
jnp.float32(1e-45) + 0  # -> 0.0

# += rebinds, doesn't mutate

# Class methods: JIT can't handle self
# Solution 1: external helper with static_argnums
# Solution 2: register as pytree
```

---

## DEBUGGING

```python
# Print
jax.debug.print("x={}", x)  # NOT print()
jax.debug.print("ordered: {}", x, ordered=True)  # preserve order

# NaN/Inf
jax.config.update("jax_debug_nans", True)
# OR
from jax.experimental import checkify
checked = checkify.checkify(f, errors=checkify.float_checks)
err, result = checked(x)
err.throw()

# Disable JIT
jax.config.update("jax_disable_jit", True)

# Inspect
jax.make_jaxpr(f)(x)

# Conditional breakpoint
def break_if_nan(x):
    jax.lax.cond(jnp.isfinite(x).all(),
                 lambda _: None,
                 lambda _: jax.debug.breakpoint(),
                 None)
```

---

## PATTERNS

```python
# Train step
@jax.jit
def train_step(params, opt_state, batch):
    def loss_fn(p):
        return jnp.mean((model(p, batch['x']) - batch['y'])**2)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

# Data generator (NumPy rng, NOT JAX)
def data_stream(images, labels, batch_size, rng=None):
    if rng is None: rng = np.random.RandomState(0)
    n, nb = len(images), len(images)//batch_size
    while True:
        perm = rng.permutation(n)
        for i in range(nb):
            idx = perm[i*batch_size:(i+1)*batch_size]
            yield images[idx], labels[idx]

# Loss/accuracy
def loss(params, batch):
    x, y = batch
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(model(params, x)) * y, axis=1))

def accuracy(params, batch):
    x, y = batch
    return jnp.mean(jnp.argmax(model(params, x), 1) == jnp.argmax(y, 1))

# Memory: donate_argnums
@partial(jax.jit, donate_argnums=0)
def update(params, grads, lr=0.01):
    return jax.tree.map(lambda p,g: p-lr*g, params, grads)
```

---

## PERFORMANCE

1. **Fuse**: Single large jit > many small
2. **Vectorize**: vmap > Python loops
3. **scan**: Fully differentiable, memory efficient
4. **Benchmark**: Use `.block_until_ready()` (async)
5. **Check**: `jax.make_jaxpr(f)(x)`

---

## QUICK REF

| Task | Function |
|------|----------|
| Compile | `jax.jit` |
| Grad | `jax.grad`, `jax.value_and_grad` |
| Vectorize | `jax.vmap` |
| Parallel | `jax.pmap` |
| If | `lax.cond`, `jnp.where` |
| Loop | `lax.scan`, `lax.fori_loop` |
| Random | `jax.random.key`, `.split` |
| Tree | `jax.tree.map`, `.leaves` |
| Debug | `jax.debug.print` |
| Custom | `custom_jvp`, `custom_vjp` |
| NaN | `checkify`, `jax_debug_nans` |

---

## CHECKLIST

- [ ] Pure functions (no side effects)
- [ ] Type annotations: `ArrayLike` in, `Array` out
- [ ] Docstrings: Args/Returns with shapes
- [ ] Static shapes for jit
- [ ] Split PRNG keys before each use (never reuse)
- [ ] Explicit state threading
- [ ] Convert lists/tuples to arrays
- [ ] No closures with mutable variables
- [ ] No iterators
- [ ] Guard operations for NaN-safe grads (`jnp.where` double-guard)
- [ ] Define jit functions once (not in loops)

---

**Think functional. Thread state. Pure functions. Transformations compose.**
