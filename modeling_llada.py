# 63 行处修改
from .SparseD_utils import create_block_mask_cached, PROBE, create_attention_block_mask

# 674 行attention替换成下列内容
    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        SparseD_param: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        # 1. 常规处理（Norm, Reshape, Rotary）- 保持原样，不要动
        B, T, C = q.size()
        dtype = k.dtype

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # Reshape: [B, nh, T, hs]
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]

        if self.config.rope:
            q, k = self.rotary_emb(q, k)

        if attention_bias is not None:
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )

        # ================= [修正后的探针位置] =================
        # 此时 q, k 形状为 [B, nh, T, hs]，切片操作才是对 Head 操作
        PROBE_STEPS = list(range(0, 129, 4)) # [20, 64, 100]
        PROBE_LAYERS = list(range(0, 31, 4)) # [0, 15, 31]

        if SparseD_param is not None and \
           SparseD_param.get("now_step") in PROBE_STEPS and \
           self.layer_id in PROBE_LAYERS:
            try:
                import os
                step = SparseD_param["now_step"]
                os.makedirs("debug_data", exist_ok=True)
                
                # 计算 Head 0 的 Attention Map
                # q[:, 0:1] -> [B, 1, T, hs]
                d_head = q.size(-1)
                scale = 1.0 / math.sqrt(d_head)
                
                # [B, 1, T, T]
                scores = torch.matmul(q[:, 0:1], k[:, 0:1].transpose(-2, -1)) * scale
                attn_probs = torch.softmax(scores, dim=-1)
                
                # 保存 [T, T]
                save_path = f"debug_data/attn_layer{self.layer_id}_step{step}.pt"
                torch.save(attn_probs[0, 0].detach().cpu().half(), save_path)
                print(f"--> [Probe] Layer {self.layer_id} Step {step} saved. Shape: {attn_probs[0, 0].shape}")
            except Exception as e:
                print(f"Probe Error: {e}")
        # ====================================================

        # 2. 注意力计算逻辑 (修正了分支判断和解包位置)
        # 如果是 Origin 模式 或 探针模式 -> 走全量 Attention
        if SparseD_param is None or SparseD_param.get("is_probe", False):
            att = self._scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0 if not self.training else self.config.attention_dropout,
                is_causal=False,
            )
        else:
            # 只有进入 SparseD 模式才解包参数，避免 Origin 模式报错 KeyError
            now_step, whole_steps, new_generation = SparseD_param['now_step'], SparseD_param['whole_steps'], SparseD_param['new_generation']
            skip, select, block_size = SparseD_param['skip'], SparseD_param['select'], SparseD_param['block_size']

            if now_step == 0:
                self.fine_mask = None
                self.last = None
                self.block_mask = None
            
            end_time = int(whole_steps*skip)+1
            if now_step <= end_time:
                # SparseD 的前 20% 也是全量计算，但需要顺便计算 Mask
                if now_step==end_time:
                    query_states, key_states, value_states = q, k, v
                    if self.fine_mask is None:
                        bsz, num_heads, q_len, kv_len = query_states.size(0), query_states.size(1), query_states.size(2), key_states.size(2)
                        self.fine_mask = torch.zeros((bsz, num_heads, (q_len+block_size-1)//block_size, (kv_len+block_size-1)//block_size), dtype=torch.bool, device=query_states.device)
                        for idx in range((q_len+block_size-1)//block_size):
                            if q_len - idx*block_size <= new_generation or idx==(q_len+block_size-1)//block_size-1:
                                if self.last is None: self.last = idx
                            query_states_reduce = query_states[:, :, idx*block_size:(idx+1)*block_size]
                            attn_weights = torch.matmul(query_states_reduce, key_states.transpose(2, 3)) / math.sqrt(num_heads)
                            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                            fine_mask = create_attention_block_mask(attn_weights, block_size=block_size, keep_ratio=select) 
                            self.fine_mask[:, :, idx:idx+1, :] = fine_mask[:, : :1, :]
                        self.fine_mask[:, :, :, self.last:] = False
                    
                    if self.block_mask is None:
                        bsz, num_heads, q_len, kv_len = query_states.size(0), query_states.size(1), query_states.size(2), key_states.size(2)
                        key_states_reduce = key_states[:, :, self.last*block_size:, :]
                        for idx in range((q_len+block_size-1)//block_size):
                            query_states_reduce = query_states[:, :, idx*block_size:(idx+1)*block_size]
                            attn_weights = torch.matmul(query_states_reduce, key_states_reduce.transpose(2, 3)) / math.sqrt(num_heads)
                            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                            fine_mask = create_attention_block_mask(attn_weights, block_size=block_size, keep_ratio=select) 
                            self.fine_mask[:, :, idx:idx+1, self.last:] = torch.logical_or(self.fine_mask[:, :, idx:idx+1, self.last:], fine_mask[:, : :1, :])
                        new_mask = customize_mask(self.fine_mask, block_size=block_size)
                        self.block_mask = create_block_mask_cached(new_mask, bsz, num_heads, q_len, kv_len, device=query_states.device, _compile=True)
                att = self._scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=0.0 if not self.training else self.config.attention_dropout,
                    is_causal=False,
                )
            else:
                att = flex_attn(q, k, v, block_mask=self.block_mask)
        
        att = att.transpose(1, 2).contiguous().view(B, T, C)
        return self.attn_out(att), present
