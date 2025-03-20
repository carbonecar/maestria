from jax import random
import timeit
import jax.numpy as jnp
import jax


def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(5.0)
print(selu(x))

key = random.key(1701)
x = random.normal(key, (1_000_000,))


x = jnp.arange(5)
isinstance(x, jax.Array)

print(timeit.timeit(lambda: selu(x).block_until_ready(), number=100))

print(x)