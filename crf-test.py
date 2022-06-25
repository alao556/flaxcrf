import jax
from flaxcrf import *

num_labels = 5
batch_first = True

rng = jax.random.PRNGKey(1024)
emission = jnp.ones([2, 10, num_labels])
mask = jnp.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
tags = jnp.array([[2, 3, 2, 3, 2, 3, 2, 3, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 0]])

crf = CRF(num_labels, batch_first)
params = crf.init(rng, emission, tags, mask)
decoded = crf.decode(params, emission, mask)
print("test successfully")
