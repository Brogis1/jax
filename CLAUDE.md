# CLAUDE.md - JAX Code Generation Specification

> JAX agent instructions. All code MUST be transformation-compatible.
> Paradigm: Functional, NOT object-oriented. Pure functions ONLY.

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

## TYPE ANNOTATIONS

### Required Types
```python
from jax import Array                    # JAX array output type
from jax.typing import ArrayLike         # Input: accepts Array, np.ndarray, Python scalars
from jax.typing import DTypeLike         # dtype specification
from collections.abc import Sequence     # For sequences
from typing import Callable, NamedTuple  # Functions, structured returns
```

### Function Signatures
```python
# ALWAYS annotate public functions
def linear(x: ArrayLike, w: ArrayLike, b: ArrayLike) -> Array:
    """Single linear layer."""
    return jnp.dot(x, w) + b

# Use ArrayLike for inputs (flexible), Array for outputs (concrete)
def normalize(x: ArrayLike, axis: int = -1) -> Array:
    return x / jnp.linalg.norm(x, axis=axis, keepdims=True)

# Multiple returns: use tuple annotation
def split_heads(x: ArrayLike, num_heads: int) -> tuple[Array, ...]:
    ...

# Pytree params: use TypeVar or dict annotation
Params = dict[str, Array]

def forward(params: Params, x: ArrayLike) -> Array:
    ...
```

### Shape Documentation
```python
def attention(
    query: ArrayLike,   # (batch, seq_len, d_model)
    key: ArrayLike,     # (batch, seq_len, d_model)
    value: ArrayLike,   # (batch, seq_len, d_model)
) -> Array:             # (batch, seq_len, d_model)
    """Scaled dot-product attention.

    Args:
        query: Query tensor of shape (batch, seq_len, d_model).
        key: Key tensor of shape (batch, seq_len, d_model).
        value: Value tensor of shape (batch, seq_len, d_model).

    Returns:
        Attention output of shape (batch, seq_len, d_model).
    """
```

### Callable Types
```python
from collections.abc import Callable

# Function arguments
def apply_fn(
    fn: Callable[[Array], Array],
    x: ArrayLike,
) -> Array:
    return fn(jnp.asarray(x))

# Loss function signature
LossFn = Callable[[Params, ArrayLike, ArrayLike], Array]

def train_step(
    loss_fn: LossFn,
    params: Params,
    x: ArrayLike,
    y: ArrayLike,
) -> tuple[Params, Array]:
    ...
```

---

## DOCSTRINGS

### Format: Google Style (Concise)
```python
def softmax(x: ArrayLike, axis: int = -1) -> Array:
    """Compute softmax along axis.

    Args:
        x: Input logits.
        axis: Axis for softmax computation. Default: -1.

    Returns:
        Softmax probabilities summing to 1 along axis.

    Example:
        >>> softmax(jnp.array([1.0, 2.0, 3.0]))
        Array([0.09, 0.24, 0.67], dtype=float32)
    """
    x_max = jnp.max(x, axis=axis, keepdims=True)
    exp_x = jnp.exp(x - x_max)
    return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)
```

### Docstring Rules
- First line: imperative verb, one sentence, no period if short
- Args: param name, then description. Include shape for arrays.
- Returns: describe output. Include shape.
- Example: optional, use `>>>` doctest format
- Note transformation compatibility if non-obvious

### Transformation Notes in Docstrings
```python
def sample(key: Array, logits: ArrayLike) -> Array:
    """Sample from categorical distribution.

    Args:
        key: PRNG key. Must be split before each call.
        logits: Unnormalized log probabilities, shape (..., num_classes).

    Returns:
        Sampled indices, shape (...).

    Note:
        JIT-compatible. PRNG key must be passed explicitly.
    """
```

---

## PURE FUNCTIONS

### Rules
- NO global state mutation
- NO print() in jitted code (use jax.debug.print)
- NO Python iterators in transformed code
- NO mutable default arguments
- ALL inputs through parameters, ALL outputs through returns

### DO / DON'T
```python
# DON'T: side effects
counter = 0
def bad(x):
    global counter
    counter += 1      # Mutates global!
    print(x)          # Side effect!
    return x

# DO: pure function
def good(x: ArrayLike, counter: int) -> tuple[Array, int]:
    return jnp.asarray(x), counter + 1
```

---

## ARRAY IMMUTABILITY

### Functional Updates
```python
# DON'T: mutation
x[0] = 1.0                    # Error in JAX!

# DO: functional update
x = x.at[0].set(1.0)          # Set single element
x = x.at[1:3].set(values)     # Set slice
x = x.at[idx].add(1.0)        # Add to element
x = x.at[mask].multiply(2.0)  # Multiply where mask
x = x.at[i, j].get()          # Safe indexing with mode control
```

---

## JIT COMPILATION

### Static Shapes Required
```python
# DON'T: dynamic output shape
@jax.jit
def bad(x: ArrayLike) -> Array:
    return x[x > 0]  # Shape depends on values!

# DO: static shapes
@jax.jit
def good(x: ArrayLike) -> Array:
    return jnp.where(x > 0, x, 0.0)  # Same shape always
```

### Static Arguments
```python
# For value-dependent control flow
@partial(jax.jit, static_argnames=['num_layers'])
def forward(
    params: Params,
    x: ArrayLike,
    num_layers: int,  # Static: causes recompilation on change
) -> Array:
    for i in range(num_layers):  # OK: num_layers is static
        x = layer(params[i], x)
    return x
```

### JIT Rules
- No value-dependent Python if/while
- Shape-dependent control OK: `if x.ndim == 2`
- Use static_argnums sparingly (recompilation cost)
- Don't jit inside loops with lambdas/partials (cache miss)

---

## AUTOMATIC DIFFERENTIATION

### grad
```python
def loss(params: Params, x: ArrayLike, y: ArrayLike) -> Array:
    """Must return scalar for grad."""
    pred = forward(params, x)
    return jnp.mean((pred - y) ** 2)

# Gradient w.r.t. first arg (default)
grads = jax.grad(loss)(params, x, y)

# Multiple args
grads = jax.grad(loss, argnums=(0, 1))(params, x, y)

# Value and grad together (efficient)
loss_val, grads = jax.value_and_grad(loss)(params, x, y)

# Composable
second_deriv = jax.grad(jax.grad(f))
```

### Jacobian/Hessian
```python
# For vector-valued outputs
J = jax.jacobian(f)(x)

# Forward-mode: efficient for tall Jacobians (few outputs)
J = jax.jacfwd(f)(x)

# Reverse-mode: efficient for wide Jacobians (few inputs)
J = jax.jacrev(f)(x)

# Hessian
H = jax.hessian(f)(x)  # = jacfwd(jacrev(f))
```

---

## VECTORIZATION (vmap)

```python
def process_single(x: ArrayLike, params: Params) -> Array:
    """Process single example."""
    return jnp.dot(params['w'], x) + params['b']

# Batch over first axis of x, broadcast params
process_batch = jax.vmap(process_single, in_axes=(0, None))

# Control axes
batched = jax.vmap(f, in_axes=(0, 1), out_axes=1)

# Nested vmap for multiple batch dims
double_batched = jax.vmap(jax.vmap(f))
```

### vmap Rules
- in_axes: batch dim per arg (None = broadcast)
- out_axes: where batch dim appears in output
- Nest for multiple batch dimensions

---

## CONTROL FLOW (JIT-Compatible)

### Prefer lax.scan (Fully Differentiable)
```python
def rnn_step(carry: Array, x: ArrayLike) -> tuple[Array, Array]:
    """Single RNN step.

    Args:
        carry: Hidden state.
        x: Input at timestep.

    Returns:
        (new_carry, output) tuple.
    """
    new_carry = jnp.tanh(carry + x)
    return new_carry, new_carry

final_state, outputs = jax.lax.scan(rnn_step, init_state, inputs)
```

### Conditional
```python
# lax.cond: both branches traced
result = jax.lax.cond(
    pred,                    # Boolean scalar
    lambda x: x + 1,         # True branch
    lambda x: x - 1,         # False branch
    operand,                 # Input
)

# jnp.where: elementwise, no function call
result = jnp.where(mask, x, y)

# lax.switch: multiple branches
result = jax.lax.switch(index, [fn0, fn1, fn2], operand)
```

### Loops
```python
# lax.fori_loop: fixed iterations
result = jax.lax.fori_loop(
    0, n,                        # Start, stop
    lambda i, val: val + i,      # Body(i, carry) -> carry
    init_val,
)

# lax.while_loop: dynamic termination
result = jax.lax.while_loop(
    lambda state: state[0] < 10,  # Condition
    lambda state: (state[0] + 1, state[1] * 2),  # Body
    (0, 1.0),                     # Initial state
)
```

### Compatibility Table
| Construct | jit | grad (rev) |
|-----------|-----|------------|
| Python if | Static only | OK |
| Python for | Unrolled | OK |
| lax.cond | OK | OK |
| lax.scan | OK | **OK** |
| lax.fori_loop | OK | fwd only* |
| lax.while_loop | OK | fwd only |

*fori_loop: rev-mode OK if bounds are static literals.

---

## RANDOM NUMBERS

### Explicit Key Threading
```python
def init_params(key: Array, dims: Sequence[int]) -> Params:
    """Initialize parameters with PRNG key.

    Args:
        key: PRNG key. Consumed by this function.
        dims: Layer dimensions.

    Returns:
        Initialized parameters.
    """
    params = []
    for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
        key, k1, k2 = jax.random.split(key, 3)
        params.append({
            'w': jax.random.normal(k1, (d_in, d_out)) * 0.01,
            'b': jax.random.zeros(k2, (d_out,)),
        })
    return params
```

### PRNG Rules
- Create: `key = jax.random.key(seed)`
- Split before use: `key, subkey = jax.random.split(key)`
- Never reuse keys (same key = same output)
- Pass key as first argument by convention

---

## PYTREES

### Nested Structures
```python
Params = dict[str, dict[str, Array]]  # Type alias

params: Params = {
    'layer1': {'w': w1, 'b': b1},
    'layer2': {'w': w2, 'b': b2},
}

# Tree operations
shapes = jax.tree.map(lambda x: x.shape, params)
zeros = jax.tree.map(jnp.zeros_like, params)
updated = jax.tree.map(lambda p, g: p - lr * g, params, grads)

# Flatten/unflatten
leaves = jax.tree.leaves(params)
```

---

## STATE MANAGEMENT

### Functional State Threading
```python
# DON'T: stateful class
class Bad:
    def __init__(self):
        self.params = init()
    def forward(self, x):
        return self.params @ x  # Hidden state!

# DO: explicit state
def init_state(key: Array) -> Params:
    return {'w': jax.random.normal(key, (10, 10))}

def forward(params: Params, x: ArrayLike) -> Array:
    return params['w'] @ x

def update(params: Params, grads: Params, lr: float = 0.01) -> Params:
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)

# Training loop threads state
params = init_state(key)
for batch in data:
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    params = update(params, grads)
```

---

## CUSTOM DERIVATIVES

### custom_jvp (Forward-Mode)
```python
@jax.custom_jvp
def log1pexp(x: ArrayLike) -> Array:
    """Numerically stable log(1 + exp(x))."""
    return jnp.log(1.0 + jnp.exp(x))

@log1pexp.defjvp
def log1pexp_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    ans = log1pexp(x)
    ans_dot = (1.0 - 1.0 / (1.0 + jnp.exp(x))) * x_dot  # Stable gradient
    return ans, ans_dot
```

### custom_vjp (Reverse-Mode)
```python
@jax.custom_vjp
def clip_gradient(x: ArrayLike, lo: float, hi: float) -> Array:
    """Identity forward, clipped gradient backward."""
    return x

def clip_fwd(x, lo, hi):
    return x, (lo, hi)  # Return residuals

def clip_bwd(res, g):
    lo, hi = res
    return (jnp.clip(g, lo, hi), None, None)

clip_gradient.defvjp(clip_fwd, clip_bwd)
```

---

## PATTERNS

### Standard Training Step
```python
@jax.jit
def train_step(
    params: Params,
    opt_state: Any,
    batch: dict[str, Array],
) -> tuple[Params, Any, Array]:
    """Single training step.

    Args:
        params: Model parameters.
        opt_state: Optimizer state.
        batch: Data batch with 'x' and 'y' keys.

    Returns:
        (updated_params, updated_opt_state, loss_value).
    """
    def loss_fn(p):
        pred = model(p, batch['x'])
        return jnp.mean((pred - batch['y']) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
```

### Batched Pairwise Distance
```python
def l2_distance(x: ArrayLike, y: ArrayLike) -> Array:
    """L2 distance between two vectors."""
    return jnp.sqrt(jnp.sum((x - y) ** 2))

# Pairwise: vmap over both
pairwise = jax.vmap(jax.vmap(l2_distance, (None, 0)), (0, None))
distances = pairwise(points, points)  # (N, N) distance matrix
```

---

## GOTCHAS

| Issue | Solution |
|-------|----------|
| Float64 disabled | `jax.config.update("jax_enable_x64", True)` |
| NaN debugging | `jax.config.update("jax_debug_nans", True)` |
| Out-of-bounds index | Returns clamped value (no error) |
| Lists as args | Convert: `jnp.array([1, 2, 3])` |
| Closure capture | Pass as arg, not closure |
| Async timing | Use `result.block_until_ready()` |

### Closure Gotcha
```python
# DON'T: captured value frozen at trace time
scale = 2.0
@jax.jit
def bad(x):
    return x * scale  # scale=2.0 forever!

# DO: pass as argument
@jax.jit
def good(x, scale):
    return x * scale
```

---

## PERFORMANCE

1. **Fuse with jit**: One big jitted fn > many small ones
2. **Vectorize**: vmap > Python loops
3. **Prefer scan**: Fully differentiable, memory efficient
4. **Avoid indexing in loops**: Vectorize instead
5. **Check jaxpr**: `jax.make_jaxpr(f)(x)` to debug tracing

---

## QUICK REFERENCE

| Task | Function |
|------|----------|
| Compile | `jax.jit` |
| Gradient | `jax.grad`, `jax.value_and_grad` |
| Vectorize | `jax.vmap` |
| Parallelize | `jax.pmap`, sharding API |
| Conditional | `lax.cond`, `jnp.where` |
| Loop | `lax.scan` (preferred), `lax.fori_loop` |
| Random | `jax.random.key`, `jax.random.split` |
| Pytree | `jax.tree.map`, `jax.tree.leaves` |
| Debug | `jax.debug.print`, `jax.make_jaxpr` |
| Custom grad | `jax.custom_jvp`, `jax.custom_vjp` |

---

## CHECKLIST

Before submitting JAX code:
- [ ] All functions pure (no side effects)
- [ ] Type annotations on public functions
- [ ] Docstrings with Args/Returns
- [ ] Shapes documented for array args
- [ ] Static shapes for jitted code
- [ ] PRNG keys split before use
- [ ] State threaded explicitly
- [ ] Transformations compose correctly

---

*Pure functions. Explicit state. Composable transformations.*
