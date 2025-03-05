from timeit import timeit

import jax.numpy as jnp
from jax import lax, jit, vmap
import jax
from jax import random
from jax.sharding import PartitionSpec as P, NamedSharding

from recurrentgemma.jax import scan


impls = []

@jax.pmap
def scan_jax(a, x):
    def body_fn(h_prev, current_inputs):
        x_t, a_t = current_inputs
        h_t = a_t * h_prev + x_t
        return h_t, h_t

    h0 = jnp.zeros_like(x[:, 0])

    scan_fn = vmap(
        lambda init, xs: lax.scan(
            body_fn,
            init=init,
            xs=xs,
            unroll=1,
        ),
        in_axes=0,
        out_axes=0,
    )
    h_last, y = scan_fn(h0, (x, a))

    return h_last, y
impls.append(('Linear(JAX)', scan_jax))


@jax.pmap
def scan_gemma(a, x):
    y, h_last = scan.lru_linear_scan(x, a, acc_float_dtype='bfloat16')
    return h_last, y
impls.append(('Linear(JAX, Gemma)', scan_gemma))


@jax.pmap
def scan_pallas(a, x):
    y, h_last = scan.lru_pallas_scan(x, a)
    return h_last, y
impls.append(('Linear(Pallas, Gemma)', scan_pallas))


if __name__ == "__main__":
    batch_size = 8
    channels = 1024
    length = 1 << 14
    rng = random.PRNGKey(0)
    a = random.normal(
        rng,
        (jax.device_count(), batch_size // jax.device_count(), length, channels),
        'bfloat16'
    )
    x = random.normal(
        rng + 1,
        (jax.device_count(), batch_size // jax.device_count(), length, channels),
        'bfloat16'
    )
    h_last_correct, y_correct = scan_jax(a, x)

    for name, f in impls:
        jax.block_until_ready(f(a, x))  # Ensure compilation
        h_last, y = f(a, x)
        assert jnp.allclose(h_last, h_last_correct, rtol=1e-1, atol=1e-1)
        assert jnp.allclose(y, y_correct, rtol=1e-1, atol=1e-1)
        def bench():
            jax.block_until_ready(f(a, x))
        t = timeit(bench, number=100) / 100
        print(f"{name}:\t{t:.4f}")
