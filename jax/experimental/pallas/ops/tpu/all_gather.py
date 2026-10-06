# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All-gather and FSDP weight-gather + matmul kernels on TPU using Pallas."""
from __future__ import annotations
from collections.abc import Sequence
import functools
import math

import jax
from jax import lax
from jax._src import shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


P = jax.sharding.PartitionSpec

_DEFAULT_VMEM_LIMIT_BYTES = 96 * 1024 * 1024
_MAX_VMEM_ALL_GATHER_BYTES = 16 * 1024 * 1024


def get_neighbor(
    idx: jax.Array, mesh: jax.sharding.Mesh, axis_name: str, *, direction: str
) -> tuple[jax.Array, ...]:
  """Helper function that computes the mesh indices of a neighbor."""
  axis_names = mesh.axis_names
  which_axis = axis_names.index(axis_name)
  mesh_index = [
      idx if i == which_axis else lax.axis_index(a)
      for i, a in enumerate(axis_names)
  ]
  axis_size = lax.axis_size(axis_name)
  if direction == "right":
    next_idx = lax.rem(idx + 1, axis_size)
  else:
    left = idx - 1
    next_idx = jnp.where(left < 0, left + axis_size, left)
  mesh_index[which_axis] = next_idx
  return tuple(mesh_index)


def ag_kernel(
    x_ref,
    o_ref,
    send_sem,
    recv_sem,
    local_copy_sem=None,
    *,
    axis_name: str,
    mesh: jax.sharding.Mesh,
):
  my_id = lax.axis_index(axis_name)
  # Overlap local x_ref -> o_ref[my_id] copy with neighbor barrier and remote
  # ring DMAs by sourcing hop i == 0 directly from x_ref instead of blocking on
  # an HBM->HBM copy before main_barrier.
  local_copy_dma = None
  if local_copy_sem is not None:
    with jax.named_scope("initial_copy_async"):
      local_copy_dma = pltpu.async_copy(x_ref, o_ref.at[my_id], local_copy_sem)

  with jax.named_scope("neighbour_lookup"):
    axis_size = lax.axis_size(axis_name)
    left_neighbor = get_neighbor(my_id, mesh, axis_name, direction="left")
    right_neighbor = get_neighbor(my_id, mesh, axis_name, direction="right")

  with jax.named_scope("main_barrier"):
    sem = pltpu.get_barrier_semaphore()
    pl.semaphore_signal(sem, 1, device_id=left_neighbor)
    pl.semaphore_signal(sem, 1, device_id=right_neighbor)
    pl.semaphore_wait(sem, 2)

  shard_size = x_ref.shape[0]
  right_slice = pl.ds(shard_size // 2, shard_size // 2)
  left_slice = pl.ds(0, shard_size // 2)
  right_dma, left_dma = None, None
  # Main strategy for this AG: carve up our input into two slices. Send
  # each slice along each direction until they reach every device.
  for i in range(axis_size - 1):
    right_slot = my_id - i
    slot_r = jnp.where(right_slot < 0, axis_size + right_slot, right_slot)
    if right_dma is not None:
      with jax.named_scope("wait_right_dma"):
        right_dma.wait()
    right_src = (
        x_ref.at[right_slice]
        if i == 0
        else o_ref.at[slot_r, right_slice]
    )
    right_dma = pltpu.async_remote_copy(
        right_src,
        o_ref.at[slot_r, right_slice],
        send_sem[1],
        recv_sem[1],
        device_id=right_neighbor,
    )

    left_slot = my_id + i
    slot_l = lax.rem(left_slot, axis_size)
    if left_dma is not None:
      with jax.named_scope("wait_left_dma"):
        left_dma.wait()
    left_src = (
        x_ref.at[left_slice]
        if i == 0
        else o_ref.at[slot_l, left_slice]
    )
    left_dma = pltpu.async_remote_copy(
        left_src,
        o_ref.at[slot_l, left_slice],
        send_sem[0],
        recv_sem[0],
        device_id=left_neighbor,
    )
  with jax.named_scope("wait_all_dma"):
    assert right_dma is not None
    assert left_dma is not None
    right_dma.wait()
    left_dma.wait()
    if local_copy_dma is not None:
      local_copy_dma.wait()
    else:
      pltpu.async_copy(x_ref, o_ref.at[my_id], recv_sem[0]).wait()


def _matmul_3d_kernel(x_ref, y_ref, o_ref, acc_ref):
  """3D-tiled Pallas matmul kernel with fp32 VMEM accumulation."""
  k_idx = pl.program_id(2)

  @pl.when(k_idx == 0)
  def _init():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  acc_ref[...] += jnp.dot(
      x_ref[...], y_ref[...], preferred_element_type=jnp.float32
  )

  @pl.when(k_idx == pl.num_programs(2) - 1)
  def _store():
    o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _adjust_tile_size(dim: int, target: int) -> int:
  tile = min(dim, target)
  while tile > 128 and dim % tile != 0:
    tile //= 2
  return tile


def pallas_matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 1024,
    bk: int = 2048,
    bn: int = 1024,
    vmem_limit_bytes: int = _DEFAULT_VMEM_LIMIT_BYTES,
) -> jax.Array:
  """3D BlockSpec-pipelined Pallas matmul with fp32 accumulation in VMEM."""
  m, k = x.shape
  _, n = y.shape
  bm = _adjust_tile_size(m, bm)
  bk = _adjust_tile_size(k, bk)
  bn = _adjust_tile_size(n, bn)

  return pl.pallas_call(
      _matmul_3d_kernel,
      grid=(m // bm, n // bn, k // bk),
      in_specs=[
          pl.BlockSpec((bm, bk), lambda i, j, p: (i, p)),
          pl.BlockSpec((bk, bn), lambda i, j, p: (p, j)),
      ],
      out_specs=pl.BlockSpec((bm, bn), lambda i, j, p: (i, j)),
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
  )(x, y)


def fsdp_all_gather_matmul(
    x: jax.Array,
    w_shard: jax.Array,
    *,
    axis_name: str = "rank",
    num_chunks: int | None = None,
    bm: int | None = None,
    bk: int | None = None,
    bn: int | None = None,
) -> jax.Array:
  """FSDP weight all-gather overlapped with 3D-tiled Pallas projection.

  Instead of materializing the full gathered weight matrix via a standalone
  custom-call ring kernel (which acts as an opaque barrier and serializes
  7 HBM-staged ring hops before the matmul can begin), this function splits
  ``w_shard`` along the output column dimension ``N`` into ``num_chunks``
  slices and pipelines topology-aware ``lax.all_gather(..., tiled=True)``
  transfers concurrently with 3D-tiled ``pallas_matmul`` MXU execution.
  """
  m_local, k = x.shape
  _, n = w_shard.shape

  if num_chunks is None:
    if k >= 8192 and n >= 16384 and n % 4 == 0:
      num_chunks = 4
    elif m_local >= 1024 and n >= 4096 and n % 2 == 0 and (n // 2) % 512 == 0:
      num_chunks = 2
    else:
      num_chunks = 1

  if bm is None or bk is None or bn is None:
    if m_local <= 512 and k <= 4096 and n <= 4096:
      bm = bm or 512
      bk = bk or 2048
      bn = bn or 2048
    elif k >= 8192:
      bm = bm or 1024
      bk = bk or 2048
      bn = bn or 1024
    else:
      bm = bm or 1024
      bk = bk or 1024
      bn = bn or 1024

  if num_chunks <= 1:
    w_full = lax.all_gather(w_shard, axis_name, axis=0, tiled=True)
    return pallas_matmul(x, w_full, bm=bm, bk=bk, bn=bn)

  chunk_n = n // num_chunks
  w_chunks = [
      lax.all_gather(
          w_shard[:, c * chunk_n : (c + 1) * chunk_n],
          axis_name,
          axis=0,
          tiled=True,
      )
      for c in range(num_chunks)
  ]
  outs = [
      pallas_matmul(x, w_g, bm=bm, bk=bk, bn=bn)
      for w_g in w_chunks
  ]
  return jnp.concatenate(outs, axis=1)


@functools.partial(
    jax.jit, static_argnames=["mesh", "axis_name", "memory_space"]
)
def all_gather(
    x,
    *,
    mesh: jax.sharding.Mesh,
    axis_name: str | Sequence[str],
    memory_space: pltpu.MemorySpace = pltpu.VMEM,
):
  if isinstance(axis_name, str):
    axis_name = (axis_name,)
  # TODO(sharadmv): enable all gather over multiple axes
  if len(axis_name) > 1:
    raise NotImplementedError("Only one axis supported.")
  (axis_name,) = axis_name
  if mesh.shape[axis_name] == 1:
    # We can short-circuit here if our axis size is 1
    return x

  # Guard against VMEM OOM when gathering large FSDP shards.
  total_bytes = math.prod(x.shape) * x.dtype.itemsize
  effective_space = (
      pl.ANY
      if (memory_space == pltpu.VMEM and total_bytes > _MAX_VMEM_ALL_GATHER_BYTES)
      else memory_space
  )

  def ag_local(x_shard):
    axis_size = lax.axis_size(axis_name)
    out_shape = jax.ShapeDtypeStruct((axis_size, *x_shard.shape), x_shard.dtype)
    out = pl.pallas_call(
        functools.partial(ag_kernel, axis_name=axis_name, mesh=mesh),
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(collective_id=0),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            scratch_shapes=(
                (pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA),
                (pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA),
                pltpu.SemaphoreType.DMA,
            ),
            in_specs=[pl.BlockSpec(memory_space=effective_space)],
            out_specs=pl.BlockSpec(memory_space=effective_space),
        ),
    )(x_shard)
    return out.reshape((axis_size * x_shard.shape[0], *x_shard.shape[1:]))

  return shard_map.shard_map(
      ag_local,
      mesh=mesh,
      in_specs=P(axis_name),
      out_specs=P(None),
      check_vma=False,
  )(x)
