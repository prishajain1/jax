# Copyright 2024 The JAX Authors.
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

"""PagedAttention TPU kernel."""

from collections.abc import Sequence
import functools
from typing import Literal

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.paged_attention import quantization_utils
import jax.numpy as jnp
import numpy as np


DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


class MultiPageAsyncCopyDescriptor:
  """Descriptor for async copy of multiple K/V pages from HBM."""

  def __init__(
      self,
      pages_hbm_ref,
      scales_pages_hbm_ref,
      vmem_buffer,
      scales_vmem_buffer,
      sem,
      page_indices,
      page_indices_start_offset,
      num_pages_to_load,
      head_index,
      *,
      page_major: bool = False,
  ):
    self._vmem_buffer = vmem_buffer
    self._scales_vmem_buffer = scales_vmem_buffer
    self._num_pages_to_load = num_pages_to_load
    self._page_major = page_major
    self._head_index = head_index
    if head_index is not None and not page_major:
      self._pages_hbm_ref = pages_hbm_ref.at[head_index]
      if scales_pages_hbm_ref is not None:
        self._scales_pages_hbm_ref = scales_pages_hbm_ref.at[head_index]
      else:
        self._scales_pages_hbm_ref = None
    else:
      self._pages_hbm_ref = pages_hbm_ref
      self._scales_pages_hbm_ref = scales_pages_hbm_ref
    self._sem = sem
    self._page_indices = page_indices
    self._page_indices_start_offset = page_indices_start_offset
    self._async_copies = [
        self._make_async_copy(i) for i in range(self._num_pages_to_load)
    ]
    if (
        self._scales_pages_hbm_ref is not None
        and self._scales_vmem_buffer is not None
    ):
      self._async_copies += [
          self._make_scales_async_copy(i)
          for i in range(self._num_pages_to_load)
      ]

  def _make_async_copy(self, i):
    page_index = self._page_indices[self._page_indices_start_offset + i]
    if self._page_major and self._head_index is not None:
      src_ref = self._pages_hbm_ref.at[page_index, :, self._head_index, :]
    else:
      src_ref = self._pages_hbm_ref.at[page_index]
    return pltpu.make_async_copy(src_ref, self._vmem_buffer.at[i], self._sem)

  def _make_scales_async_copy(self, i):
    page_index = self._page_indices[self._page_indices_start_offset + i]
    if self._page_major and self._head_index is not None:
      src_ref = self._scales_pages_hbm_ref.at[  # pyrefly: ignore[missing-attribute]
          page_index, :, self._head_index, :
      ]
    else:
      src_ref = self._scales_pages_hbm_ref.at[page_index]  # pyrefly: ignore[missing-attribute]
    return pltpu.make_async_copy(
        src_ref,
        self._scales_vmem_buffer.at[i],
        self._sem,
    )

  def start(self):
    """Starts the async copies."""
    for async_copy in self._async_copies:
      async_copy.start()

  def _maybe_dequantize(self, x, x_scale, dtype=jnp.bfloat16):
    if x_scale is None:
      return x.astype(dtype)
    return quantization_utils.from_int8(x, x_scale, dtype=dtype)

  def wait_and_get_loaded(self) -> jax.Array:
    """Wait async copies and gets the loaded buffer as a jax.Array."""
    for async_copy in self._async_copies:
      async_copy.wait()
    head_dim = self._vmem_buffer.shape[-1]
    jax_array = self._vmem_buffer[...].astype(jnp.float32)
    if self._scales_vmem_buffer is not None:
      scales_jax_array = self._scales_vmem_buffer[...].astype(jnp.float32)
    else:
      scales_jax_array = None
    jax_array = self._maybe_dequantize(jax_array, scales_jax_array)
    return jax_array.reshape(-1, head_dim)


def _paged_gqa_decode_page_major_kernel(
    lengths_ref,
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    *,
    t: int,
    bm: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
):
  """Fused GQA decode kernel for page-major [num_pages, page_size, kv_heads, d] KV caches.

  Batches all `kv_heads` and `bm` requests per Pallas program, performing the
  `(t, kv_heads, d) -> (kv_heads, t, d)` axis swap on-chip in VMEM/registers
  rather than issuing a full-pool `(2, 0, 1, 3)` transpose in HBM.
  """
  b_idx = pl.program_id(0)
  b_start = b_idx * bm
  time_idx = lax.broadcasted_iota(jnp.int32, (1, 1, t), 2)

  for i in range(bm):
    length = lengths_ref[b_start + i]
    q = q_ref[i]  # (kv_heads, g_pad, head_dim)
    k = k_ref[i].swapaxes(0, 1)  # (kv_heads, t, head_dim)
    v = v_ref[i].swapaxes(0, 1)  # (kv_heads, t, head_dim)

    scores = jnp.einsum(
        "hgd,htd->hgt", q, k, preferred_element_type=jnp.float32
    )
    if attn_logits_soft_cap is not None:
      scores = jnp.tanh(scores / attn_logits_soft_cap) * attn_logits_soft_cap

    keep = time_idx < length
    scores = jnp.where(keep, scores, mask_value)

    m = jnp.max(scores, axis=-1, keepdims=True)
    m = jnp.where(length > 0, m, 0.0)
    exp_scores = jnp.where(keep, jnp.exp(scores - m), 0.0)
    denom = jnp.sum(exp_scores, axis=-1, keepdims=True)
    probs = (exp_scores / jnp.where(denom > 0, denom, 1.0)).astype(v.dtype)

    out = jnp.einsum(
        "hgt,htd->hgd", probs, v, preferred_element_type=jnp.float32
    )
    out = jnp.where(length > 0, out, 0.0)
    o_ref[i] = out.astype(o_ref.dtype)


def paged_flash_attention_kernel(
    lengths_ref,
    page_indices_ref,
    buffer_index_ref,
    init_flag_ref,
    q_ref,
    k_pages_hbm_ref,
    k_scales_pages_hbm_ref,
    v_pages_hbm_ref,
    v_scales_pages_hbm_ref,
    o_ref,
    m_ref,
    l_ref,
    k_vmem_buffer,
    k_scales_vmem_buffer,
    v_vmem_buffer,
    v_scales_vmem_buffer,
    k_sems,
    v_sems,
    *,
    batch_size: int,
    pages_per_compute_block: int,
    pages_per_sequence: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    megacore_mode: str | None,
    page_major: bool = False,
    program_ids=(),
):
  """Pallas kernel for paged attention."""
  if program_ids:
    core_index, b, h, i = program_ids
  else:
    core_index, b, h, i = (
        pl.program_id(0),
        pl.program_id(1),
        pl.program_id(2),
        pl.program_id(3),
    )
  if page_major:
    _, page_size, num_kv_heads, _ = k_pages_hbm_ref.shape
  else:
    num_kv_heads, _, page_size, _ = k_pages_hbm_ref.shape
  bk = page_size * pages_per_compute_block
  num_cores = pl.num_programs(0)

  b_step = num_cores if megacore_mode == "batch" else 1
  b_start = core_index if megacore_mode == "batch" else 0
  h_step = num_cores if megacore_mode == "kv_head" else 1
  h_start = core_index if megacore_mode == "kv_head" else 0

  h = h * h_step + h_start
  b = b * b_step + b_start
  length = lengths_ref[b]

  def compute_block_indices(b, h, i):

    def advance_b():
      next_b = b + b_step

      def advance_to_next_non_zero_length():
        next_next_b = next_b + b_step
        return lax.fori_loop(
            lax.div(next_next_b, b_step),
            lax.div(batch_size, b_step),
            lambda _, b: jnp.where(lengths_ref[b] == 0, b + b_step, b),
            next_next_b,
        )

      return (
          lax.cond(
              jnp.logical_and(
                  next_b < batch_size,
                  lengths_ref[lax.clamp(0, next_b, batch_size - 1)] == 0),
              advance_to_next_non_zero_length,
              lambda: next_b,
          ),
          h_start,
          0,
      )

    def advance_h():
      next_h = h + h_step
      return lax.cond(next_h < num_kv_heads, lambda: (b, next_h, 0), advance_b)

    return lax.cond(i * bk < lengths_ref[b], lambda: (b, h, i), advance_h)

  def create_kv_async_copy_descriptors(b, h, i, buffer_index):
    page_offset = b * pages_per_sequence + i * pages_per_compute_block
    pages_to_load = pages_per_compute_block
    async_copy_k = MultiPageAsyncCopyDescriptor(
        k_pages_hbm_ref,
        k_scales_pages_hbm_ref,
        k_vmem_buffer.at[buffer_index],
        k_scales_vmem_buffer.at[buffer_index]
        if k_scales_vmem_buffer is not None
        else None,
        k_sems.at[buffer_index],
        page_indices_ref,
        page_offset,
        pages_to_load,
        h,
        page_major=page_major,
    )
    async_copy_v = MultiPageAsyncCopyDescriptor(
        v_pages_hbm_ref,
        v_scales_pages_hbm_ref,
        v_vmem_buffer.at[buffer_index],
        v_scales_vmem_buffer.at[buffer_index]
        if v_scales_vmem_buffer is not None
        else None,
        v_sems.at[buffer_index],
        page_indices_ref,
        page_offset,
        pages_to_load,
        h,
        page_major=page_major,
    )
    return async_copy_k, async_copy_v

  @pl.when(i * bk < length)
  def flash_attention():
    init_flag = init_flag_ref[0]
    init_flag_ref[0] = 0
    buffer_index = buffer_index_ref[0]
    next_b, next_h, next_i = compute_block_indices(b, h, i + 1)

    @pl.when(init_flag)
    def prefetch_first_block():
      async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
          b, h, i, buffer_index
      )
      async_copy_k.start()
      async_copy_v.start()

    @pl.when(i == 0)
    def init():
      m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
      l_ref[...] = jnp.zeros_like(l_ref)
      o_ref[...] = jnp.zeros_like(o_ref)

    @pl.when(next_b < batch_size)
    def prefetch_next_block():
      next_buffer_index = jnp.where(buffer_index == 0, 1, 0)
      async_copy_next_k, async_copy_next_v = create_kv_async_copy_descriptors(
          next_b, next_h, next_i, next_buffer_index
      )
      async_copy_next_k.start()
      async_copy_next_v.start()
      buffer_index_ref[0] = next_buffer_index

    async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
        b, h, i, buffer_index
    )
    q = q_ref[...].astype(jnp.float32)
    k = async_copy_k.wait_and_get_loaded()
    qk = jnp.einsum("gd,td->gt", q, k, preferred_element_type=jnp.float32)
    if attn_logits_soft_cap is not None:
      capped_qk = jnp.tanh(qk / attn_logits_soft_cap)
      qk = capped_qk * attn_logits_soft_cap

    mask = i * bk + jax.lax.broadcasted_iota(jnp.int32, qk.shape, 1) < length
    qk = qk + jnp.where(mask, 0.0, mask_value)
    m_curr = qk.max(axis=-1)

    s_curr = jnp.exp(qk - m_curr[..., None])
    m_prev, l_prev = m_ref[...], l_ref[...]
    l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
    m_curr = jax.lax.broadcast_in_dim(m_curr, m_prev.shape, (0,))
    m_next = jnp.maximum(m_prev, m_curr)
    alpha = jnp.exp(m_prev - m_next)
    beta = jnp.exp(m_curr - m_next)
    l_next = alpha * l_prev + beta * l_curr
    m_ref[...], l_ref[...] = m_next, l_next

    v = async_copy_v.wait_and_get_loaded()
    o_curr = jnp.einsum("gt,td->gd", s_curr, v)

    o_ref[...] = (
        (l_prev * alpha * o_ref[...] + beta * o_curr) / l_next
    ).astype(o_ref.dtype)


def paged_flash_attention_kernel_inline_seq_dim(
    lengths_ref,
    page_indices_ref,
    buffer_index_ref,
    init_flag_ref,
    q_ref,
    k_pages_hbm_ref,
    k_scales_pages_hbm_ref,
    v_pages_hbm_ref,
    v_scales_pages_hbm_ref,
    o_ref,
    m_ref,
    l_ref,
    k_vmem_buffer,
    k_scales_vmem_buffer,
    v_vmem_buffer,
    v_scales_vmem_buffer,
    k_sems,
    v_sems,
    *,
    batch_size: int,
    pages_per_compute_block: int,
    pages_per_sequence: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    megacore_mode: str | None,
    page_major: bool = False,
):
  core_index, b, h = pl.program_id(0), pl.program_id(1), pl.program_id(2)

  # Initialize the output HBM buffers to avoid accessing garbage memory inside
  # the kernel body below.
  m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
  l_ref[...] = jnp.zeros_like(l_ref)
  o_ref[...] = jnp.zeros_like(o_ref)

  def body(i, _):
    paged_flash_attention_kernel(
        lengths_ref,
        page_indices_ref,
        buffer_index_ref,
        init_flag_ref,
        q_ref,
        k_pages_hbm_ref,
        k_scales_pages_hbm_ref,
        v_pages_hbm_ref,
        v_scales_pages_hbm_ref,
        o_ref,
        m_ref,
        l_ref,
        k_vmem_buffer,
        k_scales_vmem_buffer,
        v_vmem_buffer,
        v_scales_vmem_buffer,
        k_sems,
        v_sems,
        batch_size=batch_size,
        pages_per_compute_block=pages_per_compute_block,
        pages_per_sequence=pages_per_sequence,
        mask_value=mask_value,
        attn_logits_soft_cap=attn_logits_soft_cap,
        megacore_mode=megacore_mode,
        page_major=page_major,
        program_ids=(core_index, b, h, i),
    )
    return ()

  page_size = (
      k_pages_hbm_ref.shape[1] if page_major else k_pages_hbm_ref.shape[-2]
  )
  bk = pages_per_compute_block * page_size

  if megacore_mode == "batch":
    num_cores = pl.num_programs(0)
    length = lengths_ref[b * num_cores + core_index]
  else:
    length = lengths_ref[b]

  lax.fori_loop(0, lax.div(length + bk - 1, bk), body, ())


@functools.partial(
    jax.jit,
    static_argnames=[
        "pages_per_compute_block",
        "attn_logits_soft_cap",
        "mask_value",
        "megacore_mode",
        "inline_seq_dim",
        "kv_pages_layout",
    ],
)
def paged_attention(
    q: jax.Array,
    k_pages: jax.Array | quantization_utils.QuantizedTensor,
    v_pages: jax.Array | quantization_utils.QuantizedTensor,
    lengths: jax.Array,
    page_indices: jax.Array,
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
    pages_per_compute_block: int,
    megacore_mode: str | None = None,
    inline_seq_dim: bool = True,
    kv_pages_layout: Literal["auto", "head_major", "page_major"] = "auto",
) -> jax.Array:
  """Paged grouped query attention.

  Args:
    q: A [batch_size, num_q_heads, head_dim] jax.Array.
    k_pages: Either a [num_kv_heads, total_num_pages, page_size, head_dim]
      (head_major) or [total_num_pages, page_size, num_kv_heads, head_dim]
      (page_major) jax.Array.
    v_pages: Either a [num_kv_heads, total_num_pages, page_size, head_dim]
      (head_major) or [total_num_pages, page_size, num_kv_heads, head_dim]
      (page_major) jax.Array.
    lengths: A i32[batch_size] jax.Array the length of each example.
    page_indices: A i32[batch_size, pages_per_sequence] jax.Array. Each entry
      should be in the range of [0, total_num_pages), indicating where to locate
      the page in `k_pages` or `v_pages`.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.
    attn_logits_soft_cap: The value used for soft capping the attention logits.
    pages_per_compute_block: how many pages to be processed in one flash
      attention block in the pallas kernel.
    megacore_mode: if set, enable megacore to parallelize the computation. Must
      be one of ['kv_head', 'batch', None]. Caveat: set this only if megacore is
      enabled, otherwise the kernel may hang. If you are not sure, leave it to
      None.
      * None: disable megacore parallelism.
      * kv_head: megacore parallelism on KV heads; requires number of KV heads
        divisible by 2.
      * batch: megacore parallelism on batch dimension; requires batch divisible
        by 2.
    inline_seq_dim: whether to fuse kernel instances along the sequence dim into
      one kernel.
    kv_pages_layout: Layout of `k_pages` and `v_pages`. `"head_major"` expects
      `[num_kv_heads, total_num_pages, page_size, head_dim]`; `"page_major"`
      expects native `[total_num_pages, page_size, num_kv_heads, head_dim]`
      without requiring a full-pool HBM transpose; `"auto"` infers layout from
      `num_q_heads` divisibility and dimension sizes.

  Returns:
    The output of attention([batch_size, num_q_heads, head_dim]).
  """
  if isinstance(k_pages, quantization_utils.QuantizedTensor):
    k_pages, k_scales_pages = k_pages.weight, k_pages.scales
    assert isinstance(k_scales_pages, jax.Array)  # For typing.
    k_scales_pages = jnp.broadcast_to(
        k_scales_pages, (*k_scales_pages.shape[:-1], k_pages.shape[-1])
    )
  else:
    k_scales_pages = None
  if isinstance(v_pages, quantization_utils.QuantizedTensor):
    v_pages, v_scales_pages = v_pages.weight, v_pages.scales
    assert isinstance(v_scales_pages, jax.Array)  # For typing.
    v_scales_pages = jnp.broadcast_to(
        v_scales_pages, (*v_scales_pages.shape[:-1], v_pages.shape[-1])
    )
  else:
    v_scales_pages = None

  batch_size, num_q_heads, head_dim = q.shape
  batch_size_paged_indices, pages_per_sequence = page_indices.shape

  if k_pages.shape != v_pages.shape:
    raise ValueError(
        f"k_pages and v_pages must have the same shape. Got {k_pages.shape} and"
        f" {v_pages.shape}"
    )

  if kv_pages_layout == "auto":
    page_major = (
        k_pages.shape[0] > num_q_heads
        and num_q_heads % k_pages.shape[2] == 0
    )
  elif kv_pages_layout == "page_major":
    page_major = True
  elif kv_pages_layout == "head_major":
    page_major = False
  else:
    raise ValueError(
        "kv_pages_layout must be one of ['auto', 'head_major', 'page_major']"
    )

  if page_major:
    _, page_size, num_kv_heads, head_dim_k = k_pages.shape
  else:
    num_kv_heads, _, page_size, head_dim_k = k_pages.shape

  if num_q_heads % num_kv_heads != 0:
    raise ValueError(
        "Number of Q heads must be divisible by number of KV heads. Got"
        f" {num_q_heads} and {num_kv_heads}."
    )
  if head_dim_k != head_dim:
    raise ValueError(
        "head_dim of Q must be the same as that of K/V. Got"
        f" {head_dim} and {head_dim_k}."
    )
  if pages_per_sequence % pages_per_compute_block != 0:
    raise ValueError(
        "pages_per_compute_block must be divisible by pages per sequence. Got"
        f" {pages_per_compute_block} and {pages_per_sequence}."
    )
  if lengths.shape != (batch_size,):
    raise ValueError("`lengths` and `q` must have the same batch size")
  if batch_size_paged_indices != batch_size:
    raise ValueError("`page_indices` and `q` must have the same batch size")
  if lengths.dtype != jnp.int32:
    raise ValueError(
        f"The dtype of `lengths` must be int32. Got {lengths.dtype}"
    )

  num_groups = num_q_heads // num_kv_heads

  # Fast path for native page-major [total_num_pages, page_size, num_kv_heads, head_dim]
  # unquantized KV caches: gather active pages directly without a full-pool
  # HBM transpose, pad the GQA group axis `g` to a multiple of 8 in native
  # dtype (avoiding the <1x128> float32 fallback when num_groups % 8 != 0),
  # and fuse all `num_kv_heads` into a batched GQA kernel with on-chip VMEM swapaxes.
  if (
      page_major
      and k_scales_pages is None
      and v_scales_pages is None
      and inline_seq_dim
      and megacore_mode is None
  ):
    total_tokens = pages_per_sequence * page_size
    bm = 2 if (batch_size >= 64 and batch_size % 2 == 0) else 1
    pad_g = ((num_groups + 7) // 8) * 8 - num_groups
    g_pad = num_groups + pad_g
    q_reshaped = q.reshape(batch_size, num_kv_heads, num_groups, head_dim)
    if pad_g > 0:
      q_pad = jnp.pad(q_reshaped, ((0, 0), (0, 0), (0, pad_g), (0, 0)))
    else:
      q_pad = q_reshaped

    k_gathered = k_pages[page_indices].reshape(
        batch_size, total_tokens, num_kv_heads, head_dim
    )
    v_gathered = v_pages[page_indices].reshape(
        batch_size, total_tokens, num_kv_heads, head_dim
    )

    out = pl.pallas_call(
        functools.partial(
            _paged_gqa_decode_page_major_kernel,
            t=total_tokens,
            bm=bm,
            mask_value=mask_value,
            attn_logits_soft_cap=attn_logits_soft_cap,
        ),
        out_shape=jax.ShapeDtypeStruct(
            (batch_size, num_kv_heads, g_pad, head_dim), q.dtype
        ),
        grid=(batch_size // bm,),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(
                (bm, num_kv_heads, g_pad, head_dim), lambda b: (b, 0, 0, 0)
            ),
            pl.BlockSpec(
                (bm, total_tokens, num_kv_heads, head_dim),
                lambda b: (b, 0, 0, 0),
            ),
            pl.BlockSpec(
                (bm, total_tokens, num_kv_heads, head_dim),
                lambda b: (b, 0, 0, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            (bm, num_kv_heads, g_pad, head_dim), lambda b: (b, 0, 0, 0)
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=96 * 1024 * 1024,
        ),
    )(lengths, q_pad, k_gathered, v_gathered)

    if pad_g > 0:
      out = out[:, :, :num_groups, :]
    return out.reshape(batch_size, num_q_heads, head_dim).astype(q.dtype)

  # TODO(dinghua): get the actual cores per chip once there's an official API.
  if megacore_mode == "kv_head":
    if num_kv_heads % 2 != 0:
      raise ValueError(
          "number of KV heads must be even when megacore_mode is 'kv_head'"
      )
    num_cores = 2
  elif megacore_mode == "batch":
    if batch_size % 2 != 0:
      raise ValueError("batch size must be even when megacore_mode is 'batch'")
    num_cores = 2
  elif megacore_mode is None:
    num_cores = 1
  else:
    raise ValueError("megacore_mode must be one of ['kv_head', 'batch', None]")

  # Pad `num_groups` to a multiple of 8 along the GQA group axis so that
  # <8x128> native VREG tiling is preserved in `q.dtype` (bfloat16) instead of
  # falling back to <1x128> float32.
  pad_g = ((num_groups + 7) // 8) * 8 - num_groups
  g_pad = num_groups + pad_g
  if pad_g > 0:
    q_grouped = q.reshape(batch_size, num_kv_heads, num_groups, head_dim)
    q = jnp.pad(
        q_grouped, ((0, 0), (0, 0), (0, pad_g), (0, 0))
    ).reshape(batch_size, num_kv_heads * g_pad, head_dim)
  if megacore_mode == "kv_head":
    q_block_spec = pl.BlockSpec(
        (None, g_pad, head_dim),
        lambda core_index, b, h, *_: (b, h * num_cores + core_index, 0),
    )
  elif megacore_mode == "batch":
    q_block_spec = pl.BlockSpec(
        (None, g_pad, head_dim),
        lambda core_index, b, h, *_: (b * num_cores + core_index, h, 0),
    )
  else:
    q_block_spec = pl.BlockSpec(
        (None, g_pad, head_dim),
        lambda core_index, b, h, *_: (b, h, 0),
    )
  q_dtype_for_kernel_launch = q.dtype

  dimension_semantics: Sequence[Literal["parallel", "arbitrary"]]
  if inline_seq_dim:
    kernel = paged_flash_attention_kernel_inline_seq_dim
    grid = (
        num_cores,
        batch_size // num_cores if megacore_mode == "batch" else batch_size,
        num_kv_heads // num_cores
        if megacore_mode == "kv_head"
        else num_kv_heads,
    )
    dimension_semantics = ("parallel", "arbitrary", "arbitrary")
  else:
    kernel = paged_flash_attention_kernel
    grid = (
        num_cores,
        batch_size // num_cores if megacore_mode == "batch" else batch_size,
        num_kv_heads // num_cores
        if megacore_mode == "kv_head"
        else num_kv_heads,
        pages_per_sequence // pages_per_compute_block,
    )
    dimension_semantics = ("parallel", "arbitrary", "arbitrary", "arbitrary")

  if k_scales_pages is not None and v_scales_pages is not None:
    in_specs = [
        q_block_spec,
        pl.BlockSpec(memory_space=pl.ANY),
        pl.BlockSpec(memory_space=pl.ANY),
        pl.BlockSpec(memory_space=pl.ANY),
        pl.BlockSpec(memory_space=pl.ANY),
    ]
    scratch_shapes = (
        pltpu.VMEM(
            (
                2,  # For double buffering during DMA copies.
                pages_per_compute_block,
                page_size,
                head_dim,
            ),
            k_pages.dtype,
        ),  # k_pages buffer
        pltpu.VMEM(
            (
                2,  # For double buffering during DMA copies.
                pages_per_compute_block,
                page_size,
                head_dim,
            ),
            k_scales_pages.dtype,
        ),  # k_scales_pages buffer
        pltpu.VMEM(
            (
                2,  # For double buffering during DMA copies.
                pages_per_compute_block,
                page_size,
                head_dim,
            ),
            v_pages.dtype,
        ),  # v_pages buffer
        pltpu.VMEM(
            (
                2,  # For double buffering during DMA copies.
                pages_per_compute_block,
                page_size,
                head_dim,
            ),
            v_scales_pages.dtype,
        ),  # v_scales_pages buffer
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
    )
  else:
    in_specs = [
        q_block_spec,
        pl.BlockSpec(memory_space=pl.ANY),
        None,
        pl.BlockSpec(memory_space=pl.ANY),
        None,
    ]
    scratch_shapes = (
        pltpu.VMEM(
            (
                2,  # For double buffering during DMA copies.
                pages_per_compute_block,
                page_size,
                head_dim,
            ),
            k_pages.dtype,
        ),  # k_pages buffer
        None,
        pltpu.VMEM(
            (
                2,  # For double buffering during DMA copies.
                pages_per_compute_block,
                page_size,
                head_dim,
            ),
            v_pages.dtype,
        ),  # v_pages buffer
        None,
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
    )

  out, _, _ = pl.pallas_call(
      functools.partial(
          kernel,
          pages_per_sequence=pages_per_sequence,
          batch_size=batch_size,
          pages_per_compute_block=pages_per_compute_block,
          mask_value=mask_value,
          attn_logits_soft_cap=attn_logits_soft_cap,
          megacore_mode=megacore_mode,
          page_major=page_major,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          # There are 4 scalars prefetched per kernel call: `lengths_ref`,
          # `page_indices_ref`, `buffer_index_ref`, `init_flag_ref`
          num_scalar_prefetch=4,
          in_specs=in_specs,
          out_specs=[
              q_block_spec,
              q_block_spec,
              q_block_spec,
          ],
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=dimension_semantics
      ),
      out_shape=[
          jax.ShapeDtypeStruct(q.shape, q_dtype_for_kernel_launch),
          jax.ShapeDtypeStruct((*q.shape[:-1], 1), jnp.float32),
          jax.ShapeDtypeStruct((*q.shape[:-1], 1), jnp.float32),
      ],
  )(
      lengths,
      page_indices.reshape(-1),
      jnp.zeros((1,), jnp.int32),  # buffer index
      jnp.ones((1,), jnp.int32),  # init flag
      q.astype(q_dtype_for_kernel_launch),
      k_pages,
      k_scales_pages,
      v_pages,
      v_scales_pages,
  )
  if pad_g > 0:
    out = out.reshape(batch_size, num_kv_heads, g_pad, head_dim)[
        :, :, :num_groups, :
    ]
  return out.reshape(batch_size, num_q_heads, head_dim).astype(q.dtype)
