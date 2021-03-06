import jax
import jax.numpy as jnp

import pytreearray as pta


def default_norm(res, t):
    if isinstance(res, jnp.ndarray):
        return jnp.sqrt(jnp.mean(jnp.abs(res) ** 2))
    else:
        return jnp.sqrt(
            jax.tree_util.tree_reduce(
                lambda x, y: x + y,
                jax.tree_map(lambda x: jnp.sum(jnp.abs(x) ** 2), res),
            )
            / res.size
        )


def calculate_error(ut, u0, u1, a, p, internalnorm, t):
    res = calculate_residuals(ut, u0, u1, a, p, internalnorm, t)
    return internalnorm(res, t)


"""
    calculate_residuals(ũ, u₀, u₁, α, ρ, internalnorm, t)

Calculate element-wise residuals
```math
\\frac{ũ}{α+\\max{|u₀|,|u₁|}*ρ}
```
"""


def calculate_residuals(ut, u0, u1, a, p, internalnorm, t):
    return ut / (a + jnp.maximum(internalnorm(u0, t), internalnorm(u1, t)) * p)
