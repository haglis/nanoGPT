import jax
import jax.numpy as jnp
import numpy as np



import numpy as np
from jax import random



from functools import partial

Explicit = jax.sharding.AxisType.Explicit
Auto = jax.sharding.AxisType.Auto

jax.config.update('jax_num_cpu_devices', 8)

mesh = jax.make_mesh((4,2), ('X', 'Y'))
for x in range(4):
    for y in range(2):
        device = mesh.devices[x, y]
        
        print(f"X={x}, Y={y}, {device}, id={device.id}")
jax.set_mesh(mesh)
import pdb;pdb.set_trace()
a = jax.device_put(jnp.arange(8 * 16.).reshape(8,16), jax.P('X', 'Y'))
b = jax.device_put(jnp.arange(16 * 4.).reshape(16,4), jax.P('Y', None))