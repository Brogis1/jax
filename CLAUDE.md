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

### += Rebinds, Does NOT Mutate
```python
# JAX behavior differs from NumPy!
jax_array = jnp.array([10, 20])
jax_array_new = jax_array
jax_array_new += 10
# jax_array_new is [20, 30], but jax_array is STILL [10, 20]!
# In NumPy both would be [20, 30] (in-place mutation)
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

### JIT Caching Gotcha
```python
# DON'T: New function object each iteration = recompile every time!
def bad_loop(x, n):
    for i in range(n):
        x = jax.jit(lambda y: y + 1)(x)  # New lambda each time!
    return x

# DON'T: partial creates new object too
def also_bad(x, n):
    for i in range(n):
        x = jax.jit(partial(add_one))(x)  # Recompiles!
    return x

# DO: Define once, reuse
@jax.jit
def add_one(x):
    return x + 1

def good_loop(x, n):
    for i in range(n):
        x = add_one(x)  # Cached after first call
    return x
```

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

### NaN Gradient Gotcha with jnp.where
```python
# DON'T: Gradients flow through BOTH branches!
def bad_log(x):
    return jnp.where(x > 0, jnp.log(x), 0.0)

jax.grad(bad_log)(0.0)  # NaN! log(0) computed even though not selected

# DO: Guard the function input too
def good_log(x):
    safe_x = jnp.where(x > 0, x, 1.0)  # Guard input
    return jnp.where(x > 0, jnp.log(safe_x), 0.0)

jax.grad(good_log)(0.0)  # 0.0, no NaN
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

# Pairwise operations
def l2_dist(x, y):
    return jnp.sqrt(jnp.sum((x - y) ** 2))

# All pairs: vmap over both args
pairwise = jax.vmap(jax.vmap(l2_dist, (None, 0)), (0, None))
distances = pairwise(points, points)  # (N, N)
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
# lax.cond: both branches traced (not lazy!)
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
| lax.cond | OK | **OK** |
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
            'b': jnp.zeros((d_out,)),
        })
    return params
```

### PRNG Rules
- Create: `key = jax.random.key(seed)`
- Split before use: `key, subkey = jax.random.split(key)`
- Never reuse keys (same key = same output)
- Pass key as first argument by convention
- Use `random.fold_in(key, i)` for deterministic per-iteration keys

### Key Reuse is WRONG
```python
# DON'T: Reusing key gives correlated/identical values!
key = jax.random.key(0)
x = jax.random.normal(key, (3,))
y = jax.random.normal(key, (3,))  # SAME as x!

# DO: Split before each use
key = jax.random.key(0)
key, k1, k2 = jax.random.split(key, 3)
x = jax.random.normal(k1, (3,))
y = jax.random.normal(k2, (3,))  # Independent from x
```

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

### Pytree Gotchas

**Tuples become nodes, not leaves:**
```python
# DON'T: shapes are tuples, become tree nodes!
a_tree = [jnp.zeros((2, 3)), jnp.zeros((3, 4))]
shapes = jax.tree.map(lambda x: x.shape, a_tree)
jax.tree.map(jnp.ones, shapes)  # Calls jnp.ones(2), jnp.ones(3), etc.!

# DO: Convert to array if needed
shapes = jax.tree.map(lambda x: jnp.array(x.shape), a_tree)
```

**None is not a leaf:**
```python
jax.tree.leaves([None, None, None])  # Returns []!

# Use is_leaf to treat None as leaf
jax.tree.leaves([None, None], is_leaf=lambda x: x is None)  # [None, None]
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

### nondiff_argnums Gotcha
```python
# Non-diff args go at START of JVP/VJP rule, regardless of position!
@partial(jax.custom_jvp, nondiff_argnums=(0, 2))
def f(fn, x, config):  # fn at 0, config at 2
    return fn(x)

@f.defjvp
def f_jvp(fn, config, primals, tangents):  # fn, config FIRST!
    (x,), (x_dot,) = primals, tangents
    ...
```

---

## ANTI-PATTERNS & GOTCHAS

### Out-of-Bounds Indexing (Silent!)
```python
# JAX does NOT raise errors for out-of-bounds!
jnp.arange(10)[11]  # Returns 9 (clamped to last), no error!

# Updates at invalid indices are silently skipped
x = jnp.arange(5)
x.at[10].set(99)  # Returns original array unchanged!

# Use explicit mode for control
x.at[11].get(mode='fill', fill_value=jnp.nan)
```

### Python Lists/Tuples as Arguments
```python
# DON'T: Each element traced separately = slow!
jnp.sum([1, 2, 3])  # Error, and bad practice

# DO: Explicit array conversion
jnp.sum(jnp.array([1, 2, 3]))
```

### Closure Capture (Frozen at Trace Time)
```python
# DON'T: Captured value frozen forever!
scale = 2.0
@jax.jit
def bad(x):
    return x * scale  # scale=2.0 baked in!

scale = 3.0
bad(jnp.array([1.0]))  # Still uses 2.0!

# DO: Pass as argument
@jax.jit
def good(x, scale):
    return x * scale
```

### Iterators Don't Work
```python
# DON'T: Iterators have state, incompatible with JAX
iterator = iter(range(10))
jax.lax.fori_loop(0, 10, lambda i, x: x + next(iterator), 0)  # Wrong result!

# DO: Use arrays
array = jnp.arange(10)
jax.lax.fori_loop(0, 10, lambda i, x: x + array[i], 0)
```

### Float64 Disabled by Default
```python
# JAX defaults to float32!
x = jnp.array([1.0, 2.0])
x.dtype  # float32, not float64!

# Enable at startup (before other JAX imports)
jax.config.update("jax_enable_x64", True)
# Or: JAX_ENABLE_X64=True environment variable
```

### Subnormal Flush-to-Zero
```python
# Subnormals flushed to zero on some backends!
subnormal = jnp.float32(1e-45)
subnormal  # 1e-45
subnormal + 0  # 0.0! Flushed to zero
```

### Class Methods with JIT
```python
# DON'T: JIT doesn't know how to handle self
class Bad:
    @jax.jit  # Error!
    def method(self, x):
        return x * self.scale

# DO Option 1: External helper function
@partial(jax.jit, static_argnums=0)
def _method(scale, x):
    return x * scale

class Good1:
    def method(self, x):
        return _method(self.scale, x)

# DO Option 2: Register as pytree (best for mutable objects)
class Good2:
    def _tree_flatten(self):
        return (self.x,), {'scale': self.scale}

    @classmethod
    def _tree_unflatten(cls, aux, children):
        return cls(*children, **aux)

jax.tree_util.register_pytree_node(Good2, Good2._tree_flatten, Good2._tree_unflatten)
```

### Type Promotion Differs from NumPy
```python
# JAX has different type promotion rules
# Binary ops may produce different dtypes than NumPy
# Unsafe casts may behave differently per backend
```

---

## DEBUGGING

### Debug Printing
```python
# DON'T: print() only runs at trace time
@jax.jit
def bad(x):
    print(x)  # Prints tracer object, only once!
    return x

# DO: jax.debug.print for runtime values
@jax.jit
def good(x):
    jax.debug.print("x = {}", x)  # Prints actual value
    return x

# Note: ordering not guaranteed unless ordered=True
jax.debug.print("a = {}", a, ordered=True)  # Preserves order (no pmap)
```

### Debug Breakpoints
```python
@jax.jit
def f(x):
    y = jnp.sin(x)
    jax.debug.breakpoint()  # Interactive debugger
    return y

# Commands: p (print), pp (pretty print), c (continue), q (quit)
```

### NaN/Inf Detection
```python
# Option 1: Global flag (easy, catches at output)
jax.config.update("jax_debug_nans", True)  # Auto-detect NaNs
jax.config.update("jax_debug_infs", True)  # Auto-detect Infs

# Option 2: checkify (works with all transforms)
from jax.experimental import checkify

@jax.jit
def f(x):
    return jnp.log(x)

checked_f = checkify.checkify(f, errors=checkify.float_checks)
err, result = checked_f(-1.0)
err.throw()  # Raises if NaN/Inf produced
```

### Disable JIT for Debugging
```python
# Temporarily disable all JIT for Python debugging
jax.config.update("jax_disable_jit", True)

# Or use environment variable: JAX_DISABLE_JIT=True

# Now can use standard Python debugging
@jax.jit  # Ignored when disabled
def f(x):
    import pdb; pdb.set_trace()  # Works!
    return x
```

### Inspect Traced Computation
```python
# See what JAX traces (jaxpr = JAX expression)
jax.make_jaxpr(f)(x)

# Check lowered HLO
lowered = jax.jit(f).lower(x)
print(lowered.as_text())
```

### Conditional Breakpoint for NaN
```python
def breakpoint_if_nan(x):
    is_finite = jnp.isfinite(x).all()
    def do_nothing(x):
        pass
    def do_break(x):
        jax.debug.breakpoint()
    jax.lax.cond(is_finite, do_nothing, do_break, x)
    return x
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

### Data Generator Pattern
```python
def data_stream(images, labels, batch_size, rng=None):
    """Infinite data stream generator.

    Args:
        images: Training images array.
        labels: Training labels array.
        batch_size: Batch size.
        rng: NumPy random state (NOT JAX key).

    Yields:
        (batch_images, batch_labels) tuples.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    num_examples = len(images)
    num_batches = num_examples // batch_size
    while True:
        perm = rng.permutation(num_examples)
        for i in range(num_batches):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            yield images[idx], labels[idx]
```

### Loss and Accuracy Functions
```python
def loss(params: Params, batch: tuple[Array, Array]) -> Array:
    """Cross-entropy loss."""
    inputs, targets = batch
    logits = predict(params, inputs)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * targets, axis=1))

def accuracy(params: Params, batch: tuple[Array, Array]) -> Array:
    """Classification accuracy."""
    inputs, targets = batch
    predicted = jnp.argmax(predict(params, inputs), axis=1)
    actual = jnp.argmax(targets, axis=1)
    return jnp.mean(predicted == actual)
```

### Memory-Efficient Update (donate_argnums)
```python
# Donate input buffer for in-place update (reduces memory)
@partial(jax.jit, donate_argnums=0)
def update(params, grads, lr=0.01):
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

---

## PERFORMANCE

1. **Fuse with jit**: One big jitted fn > many small ones
2. **Vectorize**: vmap > Python loops
3. **Prefer scan**: Fully differentiable, memory efficient
4. **Avoid indexing in loops**: Vectorize instead
5. **Check jaxpr**: `jax.make_jaxpr(f)(x)` to debug tracing
6. **Use donate_argnums**: For in-place buffer reuse
7. **Benchmark correctly**: Use `.block_until_ready()` (async dispatch)

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
| Check NaN | `checkify.checkify`, `jax_debug_nans` |

---

## CHECKLIST

Before submitting JAX code:
- [ ] All functions pure (no side effects)
- [ ] Type annotations on public functions
- [ ] Docstrings with Args/Returns/shapes
- [ ] Static shapes for jitted code
- [ ] PRNG keys split before each use
- [ ] State threaded explicitly (no hidden state)
- [ ] No list/tuple args (convert to arrays)
- [ ] No closure capture of changing values
- [ ] No iterators in JAX code
- [ ] Guarded operations for NaN-safe gradients
- [ ] JIT functions defined once, not in loops

---

*Pure functions. Explicit state. Composable transformations.*
