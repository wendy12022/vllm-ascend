#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch_npu
import os
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)
from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils import cdiv, direct_register_custom_op
from vllm.v1.attention.backends.utils import AttentionCGSupport
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         maybe_save_kv_layer_to_connector,
                                         wait_for_kv_layer_from_connector)
from vllm_ascend.compilation.acl_graph import (get_graph_params,
                                               update_graph_params_workspaces)
from vllm_ascend.ops.attention import vanilla_chunked_prefill
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ, aligned_16, is_310p, is_910a,
                               nd_to_nz_2d, nd_to_nz_spec)

from ..utils import weak_ref_tensors


class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND"

    @staticmethod
    def get_impl_cls() -> Type["AscendAttentionBackendImpl"]:
        return AscendAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["AscendMetadata"]:
        return AscendMetadata

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        if is_310p():
            return (2, num_blocks, num_kv_heads * head_size // 16, block_size,
                    16)
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_bsh_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads * head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: List[torch.Tensor],
        dst_kv_cache: List[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(
            dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(
            dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            key_caches = kv_cache[0]
            value_caches = kv_cache[1]
            key_caches[dst_indices] = key_caches[src_indices]
            value_caches[dst_indices] = value_caches[src_indices]

    @staticmethod
    def get_supported_block_size() -> list[int]:
        return [128]


class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


@dataclass
class AscendMetadata:

    # **************************** Basic Properties ************************** #
    attn_mask: Optional[torch.Tensor] = None
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    # Number of tokens excluding padding.
    num_actual_tokens: int = 0

    # The sequence length per sequence. Sequence length means the computed
    # tokens + new tokens (is None if it is a decoding).
    # (batch_size,)
    # TODO(Angazenn): The following parameters are quite redundant and
    # contains similar information (such as seq_lens seq_lens_list). We
    # should simplified these parameters once attention schema in vLLM-Ascend
    # is unified.
    seq_lens: torch.Tensor = None
    seq_lens_list: List[int] = None  # type: ignore
    actual_seq_lengths_q: List[int] = None  # type: ignore

    query_start_loc: torch.Tensor = None
    query_lens: torch.Tensor = None
    # Maximum query length in the batch (None for decoding).
    max_query_len: Optional[int] = None

    # ********************** KV Cache Related Properties ********************* #
    # Block addresses per sequence (Seq id -> list of physical block).
    # (batch_size, max_blocks_per_seq)
    block_tables: torch.Tensor = None

    # The indices of the token slots that input tokens will be stored into.
    # E.g., if `slot_mapping` is [35, 2, 17] and the block size is 16, the
    # three tokens are stored in the 3rd slot in block 2, 2nd slot in block 0,
    # and 1st slot in block 1, respectively.
    # (num_tokens,)
    slot_mapping: torch.Tensor = None

    # *************************** Other Properties *************************** #
    enable_dbo_across_dp: bool = False


class AscendAttentionMetadataBuilder:
    # Does this backend/builder support ACL Graphs for attention (default: no).
    aclgraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: ClassVar[int] = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len,
            AscendAttentionBackend.get_supported_block_size()[0])
        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"

    def reorder_batch(self, input_batch,
                      scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        model: Optional[nn.Module] = None,
    ):
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[:
                                                                       num_reqs
                                                                       + 1]
        block_table = common_attn_metadata.block_table_tensor
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]
        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]
        attn_mask = common_attn_metadata.attn_mask
        attn_state = common_attn_metadata.attn_state
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[:
                                                                       num_reqs
                                                                       + 1]

        if attn_state == AscendAttentionState.DecodeOnly and \
            common_attn_metadata.num_input_tokens > num_actual_tokens:
            padded_num_tokens = common_attn_metadata.num_input_tokens - num_actual_tokens
            seq_lens = torch.cat([
                seq_lens,
                torch.ones(padded_num_tokens,
                           dtype=seq_lens.dtype,
                           device=seq_lens.device)
            ])
            block_table_padding = torch.zeros(
                (padded_num_tokens, ) + block_table.shape[1:],
                dtype=block_table.dtype,
                device=block_table.device)
            block_table = torch.cat([block_table, block_table_padding], dim=0)
            query_start_loc_cpu = torch.cat([
                query_start_loc_cpu,
                torch.arange(query_start_loc_cpu[-1] + 1,
                             query_start_loc_cpu[-1] + padded_num_tokens,
                             dtype=query_start_loc_cpu.dtype,
                             device=query_start_loc_cpu.device)
            ])

        query_start_loc = query_start_loc_cpu.to(self.device,
                                                 non_blocking=True)

        if is_310p():
            if attn_state == AscendAttentionState.PrefillNoCache:
                mask_nz = nd_to_nz_2d(attn_mask)
                attn_mask = torch_npu.npu_format_cast(mask_nz.contiguous(),
                                                      ACL_FORMAT_FRACTAL_NZ)
            elif attn_state == AscendAttentionState.ChunkedPrefill:
                mask_nz = nd_to_nz_spec(attn_mask)
                attn_mask = torch_npu.npu_format_cast(mask_nz.contiguous(),
                                                      ACL_FORMAT_FRACTAL_NZ)

        attn_metadata = AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            query_lens=query_lens,
            seq_lens=seq_lens,
            seq_lens_list=seq_lens.tolist(),
            max_query_len=common_attn_metadata.max_query_len,
            actual_seq_lengths_q=query_start_loc_cpu[1:].tolist(),
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            attn_state=attn_state,
            enable_dbo_across_dp=common_attn_metadata.enable_dbo_across_dp)
        return attn_metadata

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
        model: Optional[nn.Module] = None,
    ):
        if attn_state == AscendAttentionState.DecodeOnly:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly state"
            )

        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str],
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes,
                                        dtype=torch.float32,
                                        device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None


    def _forward_prefill_no_cache_910a_kvgroup(self, query, key, value, attn_metadata, output, num_tokens=0):
        """Per-KV-head 分组版本：将同 KV-head 的 Q-heads 打包计算，ki.T 只算一次。

        相比逐 Q-head 版本：
        - kernel launch 数减少 n_rep 倍 (TP=8 时 32B: 8x减少, 8B: 4x减少)
        - matmul M 维度放大 n_rep 倍 → Cube 单元利用率更高
        """
        real_len = num_tokens
        if attn_metadata.attn_mask is not None:
            mask_size = attn_metadata.attn_mask.shape[-1]
            real_len = min(num_tokens, mask_size)

        q = query[:real_len]
        k = key[:real_len]
        v = value[:real_len]

        q = q.transpose(0, 1)  # [num_q_heads, N, dim]
        k = k.transpose(0, 1)  # [num_kv_heads, N, dim]
        v = v.transpose(0, 1)  # [num_kv_heads, N, dim]

        num_q_heads = q.shape[0]
        num_kv_heads = k.shape[0]
        n_rep = num_q_heads // num_kv_heads
        head_dim = self.head_size

        # 临时恢复自适应版本，排除 Q_CHUNK_KV 变化导致性能回退
        chunk_mb = int(os.environ.get('VLLM_910A_ATTN_CHUNK_MB', '600'))
        TARGET_BYTES = max(64_000_000, min(2_000_000_000, chunk_mb * 1_000_000))
        Q_CHUNK_KV = max(64, min(2048, TARGET_BYTES // max(1, n_rep * real_len * 4)))

        full_mask = attn_metadata.attn_mask if attn_metadata.attn_mask is not None else None
        orig_dtype = q.dtype

        try:
            # 预分配输出，直接写入，省掉 cat+transpose+contiguous
            final_out = torch.empty(real_len, num_q_heads, head_dim,
                                   dtype=orig_dtype, device=q.device)

            for kv_idx in range(num_kv_heads):
                ki_t = k[kv_idx:kv_idx+1].transpose(-2, -1)  # [1, dim, N] — 只算1次!
                vi = v[kv_idx:kv_idx+1]                       # [1, N, dim]

                q_start = kv_idx * n_rep
                q_end = q_start + n_rep
                qi_group = q[q_start:q_end]  # [n_rep, N, dim]

                for chunk_start in range(0, real_len, Q_CHUNK_KV):
                    chunk_end = min(chunk_start + Q_CHUNK_KV, real_len)
                    qi_chunk = qi_group[:, chunk_start:chunk_end]  # [n_rep, Qc, dim]

                    attn_chunk = torch.matmul(qi_chunk, ki_t) * self.scale

                    if full_mask is not None:
                        mask_chunk = full_mask[chunk_start:chunk_end, :real_len]
                        mask_chunk = torch.where(
                            torch.isinf(mask_chunk),
                            torch.tensor(-10000.0, device=q.device, dtype=torch.float32),
                            mask_chunk.float())
                        attn_chunk = attn_chunk.float() + mask_chunk
                        del mask_chunk
                    else:
                        attn_chunk = attn_chunk.float()

                    p_chunk = torch.softmax(attn_chunk, dim=-1).to(orig_dtype)
                    del attn_chunk

                    out_chunk = torch.matmul(p_chunk, vi)  # [n_rep, Qc, dim]
                    # 直接写入预分配的输出 (Q-heads维度已按KV-head分组)
                    final_out[chunk_start:chunk_end, q_start:q_end] = out_chunk.transpose(0, 1)
                    del p_chunk, out_chunk

            output[:real_len, :, :] = final_out
            del final_out

        except Exception as e:
            print(f"CRITICAL: Prefill Logic Failed at real_len {real_len}! Error: {e}")
            raise e

        return output[:num_tokens]

    def _forward_prefill_no_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        num_tokens=0,
    ) -> torch.Tensor:
        assert attn_metadata is not None
        assert attn_metadata.attn_mask is not None

        mask = attn_metadata.attn_mask

        if is_310p():
            # align q k v output tensors
            query = aligned_16(query)
            key = aligned_16(key)
            value = aligned_16(value)
            output = aligned_16(output)
            # do reformat in case of broadcasted tensors
            mask = mask.repeat(attn_metadata.seq_lens.size(0), 1, 1, 1)
            mask = torch_npu.npu_format_cast(mask.contiguous(),
                                             ACL_FORMAT_FRACTAL_NZ)

        torch_npu._npu_flash_attention(query=query,
                                       key=key,
                                       value=value,
                                       mask=mask,
                                       seq_len=attn_metadata.seq_lens,
                                       scale_value=self.scale,
                                       num_heads=self.num_heads,
                                       num_kv_heads=self.num_kv_heads,
                                       out=output)
        assert output is not None
        return output[:num_tokens, :, :]

    def _forward_prefill_cache_hit(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert attn_metadata is not None
        assert attn_metadata.attn_mask is not None

        compress_mask = attn_metadata.attn_mask
        batch_size = attn_metadata.query_lens.shape[0]
        block_table = attn_metadata.block_tables[:batch_size, :]

        torch_npu._npu_flash_attention_qlens(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            block_table=block_table,
            mask=compress_mask,
            seq_len=attn_metadata.query_lens,
            context_lens=attn_metadata.seq_lens,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            out=output)
        return output

    def _forward_decode_only_910a(
    self,
    query: torch.Tensor,
    attn_metadata: AscendMetadata,
    output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode attention with per-KV-head batched matmul（无 repeat_interleave）。

        Q reshape 为 [kv_h, n_rep, d], K/V 保持 [kv_h, ...]，
        一次 batched matmul 完成所有 KV-head 的 attention 计算。
        """
        batch_size = query.shape[0]
        if output is None:
            output = torch.empty_like(query)

        num_kv_heads = self.num_kv_heads
        head_size = self.head_size
        block_size = self.key_cache.shape[1]
        n_rep = self.num_heads // self.num_kv_heads
        orig_dtype = query.dtype

        for i in range(batch_size):
            seq_len = attn_metadata.seq_lens[i].item()
            block_table = attn_metadata.block_tables[i]

            num_needed_blocks = (seq_len + block_size - 1) // block_size
            curr_blocks = block_table[:num_needed_blocks].to(torch.long)

            # 从 Cache 提取 KV, 保持原始 dtype 避免额外显存
            k_cur = self.key_cache[curr_blocks].view(-1, num_kv_heads, head_size)[:seq_len]
            v_cur = self.value_cache[curr_blocks].view(-1, num_kv_heads, head_size)[:seq_len]

            # Q: [num_heads, d] → [num_kv_heads, n_rep, d]
            q_i = query[i].view(num_kv_heads, n_rep, head_size)
            # K: [seq_len, kv_h, d] → [kv_h, d, seq_len]
            k_i = k_cur.permute(1, 2, 0)
            # V: [seq_len, kv_h, d] → [kv_h, seq_len, d]
            v_i = v_cur.permute(1, 0, 2)

            # batched matmul: [kv_h, n_rep, d] × [kv_h, d, seq_len] → [kv_h, n_rep, seq_len]
            attn_scores = torch.matmul(q_i, k_i) * self.scale
            attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(orig_dtype)

            # batched matmul: [kv_h, n_rep, seq_len] × [kv_h, seq_len, d] → [kv_h, n_rep, d]
            out_i = torch.matmul(attn_probs, v_i)
            out_i = out_i.reshape(self.num_heads, head_size)

            output[i].copy_(out_i)

        return output

    def _forward_decode_only(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if is_310p():
            # seq_lens_tensor needs to be transferred to the device for 310P.
            attn_metadata.seq_lens = \
                attn_metadata.seq_lens.to(device=query.device)
        if self.sliding_window is not None and attn_metadata.seq_lens.shape[
                0] == query.size(0):
            batch_size = attn_metadata.seq_lens.shape[0]
            block_size = 128
            query = query.view(batch_size, 1, self.num_heads * self.head_size)
            key = self.key_cache
            value = self.value_cache
            if self.key_cache is not None and self.value_cache is not None:
                block_size = self.key_cache.shape[1]
                key = self.key_cache.flatten(2, 3).contiguous()
                value = self.value_cache.flatten(2, 3).contiguous()

            output, _ = torch_npu.npu_fused_infer_attention_score(
                query,
                key,
                value,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BSH",
                block_size=block_size,
                pre_tokens=self.sliding_window,
                scale=self.scale,
                block_table=attn_metadata.block_tables,
                actual_seq_lengths=[1] * len(attn_metadata.seq_lens),
                actual_seq_lengths_kv=attn_metadata.seq_lens)

            output = output.view(batch_size, self.num_heads, self.head_size)
        else:
            graph_params = get_graph_params()
            forward_context: ForwardContext = get_forward_context()
            num_tokens = query.shape[0]
            if forward_context.capturing:
                # Get workspace from cache or calculate it if not present.
                workspace = graph_params.workspaces.get(num_tokens)
                if workspace is None:
                    workspace = torch_npu._npu_paged_attention_get_workspace(
                        query=query,
                        key_cache=self.key_cache,
                        value_cache=self.value_cache,
                        num_kv_heads=self.num_kv_heads,
                        num_heads=self.num_heads,
                        scale_value=self.scale,
                        block_table=attn_metadata.block_tables,
                        context_lens=attn_metadata.seq_lens,
                        out=output)
                    update_graph_params_workspaces(num_tokens,
                                                   weak_ref_tensors(workspace))

                # Handle graph capturing mode
                stream = torch_npu.npu.current_stream()

                event = torch.npu.ExternalEvent()
                event.wait(stream)
                event.reset(stream)
                graph_params.events[num_tokens].append(event)
                graph_params.attn_params[num_tokens].append((
                    weak_ref_tensors(query),
                    weak_ref_tensors(self.key_cache),
                    weak_ref_tensors(self.value_cache),
                    self.num_kv_heads,
                    self.num_heads,
                    self.scale,
                    attn_metadata.block_tables,
                    attn_metadata.seq_lens,
                    weak_ref_tensors(output),
                ))

                torch.npu.graph_task_group_begin(stream)
                torch_npu._npu_paged_attention(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=attn_metadata.block_tables,
                    context_lens=attn_metadata.seq_lens,
                    out=output,
                    workspace=workspace)
                handle = torch.npu.graph_task_group_end(stream)
                graph_params.handles[num_tokens].append(handle)
            else:
                torch_npu._npu_paged_attention(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=attn_metadata.block_tables,
                    context_lens=attn_metadata.seq_lens,
                    out=output)
        return output

    def _forward_v1_style(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Use chunked prefill for head size 192 scenario, like deepseek
        # paged_attention_splitfuse maybe crash at such scenario.
        # TODO: vanilla path will be removed after the kernel support
        # head_size 192 scenario.
        if self.head_size == 192:
            cu_seqlen_q = [0] + attn_metadata.query_lens.tolist()
            cu_seqlen_k = [0] + attn_metadata.seq_lens.tolist()
            cu_seqlen_q = torch.tensor(cu_seqlen_q, device=query.device)
            cu_seqlen_k = torch.tensor(cu_seqlen_k, device=query.device)
            cu_seqlen_q = torch.cumsum(cu_seqlen_q, dim=0)
            cu_seqlen_k = torch.cumsum(cu_seqlen_k, dim=0)
            max_seqlen_q = torch.max(attn_metadata.query_lens)
            max_seqlen_k = torch.max(attn_metadata.seq_lens)
            vanilla_chunked_prefill(output, query, self.key_cache,
                                    self.value_cache,
                                    attn_metadata.block_tables, cu_seqlen_q,
                                    cu_seqlen_k, max_seqlen_q, max_seqlen_k,
                                    self.scale, None, True)
            return output

        # Use paged attention.
        assert attn_metadata is not None
        assert attn_metadata.attn_mask is not None

        if is_310p():
            # Do reformat in case of broadcasted tensors.
            attn_metadata.attn_mask = \
                torch_npu.npu_format_cast(attn_metadata.attn_mask.contiguous(),
                                          ACL_FORMAT_FRACTAL_NZ)
            attn_metadata.seq_lens = \
                attn_metadata.seq_lens.to(device=query.device)

        # TODO:The npu_fused_infer_attention_score op is planned to
        # be utilized in a wider range in upcoming versions.
        num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
        key = self.key_cache.view(  # type: ignore
            num_block, block_size, -1)
        value = self.value_cache.view(  # type: ignore
            num_block, block_size, -1)

        output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=attn_metadata.block_tables,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )

        return output

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        trace_flag: bool = True,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            kv_cache: shape = [key_cache, value_cache]
                      key_cache = [num_blocks, block_size,
                                   num_kv_heads, head_size]
                      value_cache = [num_blocks, block_size,
                                     num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size * seq_len, num_heads, head_size]
        """
        num_tokens = query.shape[0]
        use_kv_cache_int8 = len(
            kv_cache) > 0 and kv_cache[0].dtype == torch.int8
        if output is None:
            output = torch.empty(num_tokens,
                                 self.num_heads,
                                 self.head_size,
                                 dtype=query.dtype,
                                 device=query.device)
        ori_output = output
        if trace_flag:
            torch.ops.vllm.unified_ascend_attention_with_output(
                query=query,
                key=key,
                value=value,
                output=output,
                layer_name=layer.layer_name)

        elif hasattr(layer, 'quant_method') and use_kv_cache_int8:
            output = layer.quant_method.apply(layer, query, key, value,
                                              kv_cache, attn_metadata,
                                              self.attn_type, self.scale,
                                              output)

        else:
            if attn_metadata is None:
                return output.view(num_tokens, self.hidden_size).fill_(0)
            num_actual_tokens = attn_metadata.num_actual_tokens
            assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
            attn_type = self.attn_type
            if attn_type != AttentionType.DECODER and attn_type != AttentionType.ENCODER_ONLY:
                raise NotImplementedError("Encoder/decoder cross-attention "
                                          "are not implemented for "
                                          "PallasAttentionBackendImpl")
            # View q k v to BSH.
            query = query.view(-1, self.num_heads, self.head_size)
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
            # TODO: Remove this contiguous in the future.
            value = value.contiguous()
            
            if len(kv_cache) > 1:
                if is_910a():
                    if self.key_cache is None:
                        self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
                    
                    current_block_size = getattr(attn_metadata, 'block_size', self.key_cache.shape[1])
                    key_fixed = key.contiguous()
                    value_fixed = value.contiguous()                    
                    
                    # 910A IndexPut 不支持，改为：slot_mapping 一次性搬 CPU → 按 block 分组 → 连续 slot 批量 copy_
                    s_map_cpu = attn_metadata.slot_mapping.flatten().cpu().tolist()

                    block_groups = {}
                    for i in range(num_actual_tokens):
                        slot = s_map_cpu[i]
                        if slot < 0:
                            continue
                        b_idx = slot // current_block_size
                        s_idx = slot % current_block_size
                        block_groups.setdefault(b_idx, []).append((i, s_idx))

                    for b_idx, tokens in block_groups.items():
                        tokens.sort(key=lambda x: x[1])
                        run_start = 0
                        for j in range(1, len(tokens) + 1):
                            if j == len(tokens) or tokens[j][1] != tokens[j - 1][1] + 1:
                                t0 = tokens[run_start][0]
                                t1 = tokens[j - 1][0] + 1
                                s0 = tokens[run_start][1]
                                s1 = tokens[j - 1][1] + 1
                                self.key_cache[b_idx, s0:s1].copy_(key_fixed[t0:t1])
                                self.value_cache[b_idx, s0:s1].copy_(value_fixed[t0:t1])
                                run_start = j

                else:
                    if self.key_cache is None:
                        self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
                    slots = attn_metadata.slot_mapping
                    torch_npu._npu_reshape_and_cache(
                        key=key[:num_actual_tokens],
                        value=value[:num_actual_tokens],
                        key_cache=self.key_cache,
                        value_cache=self.value_cache,
                        slot_indices=slots)
            
            if is_910a():        
                # V0-Style scheduler situation.
                if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
                    # GQA (n_rep>1) 时用 per-KV-head 分组: 减少 kernel launch + 提升 Cube 利用率
                    if self.num_heads > self.num_kv_heads:
                        output = self._forward_prefill_no_cache_910a_kvgroup(
                                query, key, value, attn_metadata, output, num_tokens)
                    else:
                        output = self._forward_prefill_no_cache_910a(
                            query, key, value, attn_metadata, output, num_tokens)
                elif attn_metadata.attn_state == \
                    AscendAttentionState.PrefillCacheHit:
                    output = self._forward_prefill_cache_hit(
                        query, attn_metadata, output)
                elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                    output = self._forward_decode_only_910a(query, attn_metadata,
                                                    output)
                # Normal V1 situation.
                else:
                    output = self._forward_v1_style(query, attn_metadata, output)        
                
            else :        
                if attn_type == AttentionType.ENCODER_ONLY:
                    cum_seq_len = attn_metadata.query_start_loc[1:].tolist()
                    attn_out = torch_npu.npu_fusion_attention(
                        query,
                        key,
                        value,
                        head_num=self.num_heads,
                        input_layout="TND",
                        scale=self.scale,
                        sparse_mode=4,
                        atten_mask=attn_metadata.attn_mask,
                        pre_tockens=attn_metadata.max_query_len,
                        next_tockens=attn_metadata.max_query_len,
                        actual_seq_qlen=cum_seq_len,
                        actual_seq_kvlen=cum_seq_len,
                    )
                    output = attn_out[0]
                # V0-Style scheduler situation.
                elif attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
                    output = self._forward_prefill_no_cache(
                        query, key, value, attn_metadata, output, num_tokens)
                elif attn_metadata.attn_state == \
                    AscendAttentionState.PrefillCacheHit:
                    output = self._forward_prefill_cache_hit(
                        query, attn_metadata, output)
                elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                    output = self._forward_decode_only(query, attn_metadata,
                                                    output)
                # Normal V1 situation.
                else:
                    # npu_fused_infer_attention_score does not support cases
                    # where query.shape[0] != attn_metadata.query_start_loc[-1].
                    # Thus we need unpad it here.
                    num_tokens = attn_metadata.query_start_loc[-1]
                    query = query[:num_tokens]
                    output = self._forward_v1_style(query, attn_metadata, output)

        # to make in-place change to the output tensor
        if hasattr(layer, 'quant_method') and use_kv_cache_int8:
            output = output.view(num_tokens, self.num_heads, self.head_size)
        ori_output[:num_tokens, :, :] = output[:num_tokens, :, :]
        return output.view(num_tokens, self.hidden_size)


def unified_ascend_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    wait_for_kv_layer_from_connector(layer_name)
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    self.impl.forward(self,
                      query,
                      key,
                      value,
                      kv_cache,
                      attn_metadata,
                      output,
                      trace_flag=False)
    maybe_save_kv_layer_to_connector(layer_name, kv_cache)
    return


def unified_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_ascend_attention_with_output",
    op_func=unified_ascend_attention_with_output,
    mutates_args=["output"],
    fake_impl=unified_attention_with_output_fake,
    dispatch_key="PrivateUse1",
)
