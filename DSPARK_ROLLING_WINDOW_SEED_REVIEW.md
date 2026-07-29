<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DSpark 128 Rolling Window 与 Disaggregated Seed Handoff Review

## 1. Review 范围与结论

本文分两部分：

1. 只依据 DSpark paper、Hugging Face 参考实现和 DeepSpec，总结 128 rolling
   window 的语义、生成方式及其在 disaggregated serving 中的合理归属。
2. 审查当前 TensorRT-LLM（TRT-LLM）的 prefill-to-decode seed handoff，判断
   `seed=on` 是否符合上述语义，并解释为什么实验中 aggregated、disaggregated
   seed-on 和 seed-off 的 acceptance length（AL）非常接近。

核心结论如下：

- 128 rolling window 是 DSpark 各 draft attention layer 的、按 token 位置保存的
  target-hidden-derived KV cache。它不是 128-token draft，不是 Markov head 的历史，
  也不是 confidence head 的状态。
- 该 window 的生成不依赖 Markov、confidence 或 LM head。它由主模型若干目标层的
  hidden states，经 DSpark 的 projection、KV projection、normalization 和 RoPE 后写入。
- 在 disaggregated serving 中，由 prefill 端填满 prompt 尾部 window 并传给 decode，
  在语义上最接近 aggregated execution。prefill 不需要提前执行第一轮完整 drafter；
  decode 端拿到 seed 后再执行第一轮 draft 即可。
- 当前 TRT-LLM `seed=on` 的主数据路径符合预期：prefill 生成 window 和逻辑长度，
  传输端携带两者，decode 在分配 request slot 后、第一次 draft forward 前应用，
  attention 随后会实际读取该 window。没有发现会使 seed 错位、应用过晚或完全不生效的
  明显实现错误。
- aggregated/on/off 的 AL 接近是合理现象，不能单凭这一点判断 seed 没有生效。最重要的
  原因是：
  1. seed-off 的空槽由 `ctx_len` mask 排除，并不会把 128 个零 KV 当作有效上下文；
  2. decode 每轮仍直接获得一个包含完整 prompt 信息的最新 target hidden；
  3. 每轮接受的中间 target tokens 都会回填 window，通常不是每轮只恢复一个槽；
  4. 128 window 主要补充高分辨率的短程上下文，可能不是该 checkpoint draft 质量的
     主导信号；
  5. 当前 request-level AL 对短 OSL 存在末轮截断/整轮计数偏差，且现有样本量较小。
- 但当前仓库存在一个重要的可复现性问题：实验脚本使用
  `TRTLLM_DSPARK_DISABLE_SEED=1` 表示 seed-off，而当前代码中没有该环境变量的消费者。
  历史实验若是在临时 kill-switch patch 尚未回退时运行，结论仍可能有效；但以当前
  checkout 重跑时，所谓 seed-off 实际也会走 seed-on。最终结论应在恢复一个可验证的
  seed-off 控制后再确认。

因此，本 review 对当前状态的判断是：

> `seed=on` 的实现逻辑基本正确；现有 AL 结果支持“传输 prompt rolling window 对测试
> workload 的收益很小”，但还不足以证明 seed 在模型级没有作用。需要一次 first-draft
> logits/attention 的配对差分实验，以及真正可复现的 seed-off A/B，才能关闭验证。

## 2. 参考资料

机制分析主要参考：

- [DSpark paper](https://arxiv.org/abs/2607.05147)
- [DeepSeek-V4-Pro-DSpark model card and files](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark)
- [Hugging Face inference reference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark/tree/main/inference)
- [DeepSpec](https://github.com/deepseek-ai/DeepSpec)

TRT-LLM 实现分析主要参考：

- `tensorrt_llm/_torch/speculative/dspark.py`
- `tensorrt_llm/_torch/models/modeling_dspark.py`
- `tensorrt_llm/_torch/models/dspark/attention.py`
- `tensorrt_llm/_torch/disaggregation/auxiliary.py`
- `tensorrt_llm/_torch/disaggregation/transfer.py`
- `tensorrt_llm/_torch/disaggregation/transceiver.py`
- `tensorrt_llm/_torch/pyexecutor/py_executor.py`
- `tests/unittest/_torch/speculative/test_dspark_disagg_seed.py`
- `DSPARK_DISAGG_SEED_HANDOFF.md`

---

## Part I：不依赖 TRT-LLM 实现的 DSpark 机制分析

## 3. DSpark 一轮 speculative decoding 做什么

DSpark 将 drafter 分成两个职责不同的部分：

1. **Parallel backbone**
   - 一次计算产生一个最大长度为 `gamma` 的 draft block 的 base logits。
   - DeepSeek-V4-Pro-DSpark 的 `dspark_block_size` 为 5。
   - block 内的候选位置由 backbone 并行计算，而不是像传统 autoregressive drafter
     一样逐 token 串行执行完整网络。

2. **Lightweight sequential head**
   - production 配置使用 Markov head。
   - 它根据前一个已采样 draft token，为下一个位置的 base logits 增加低秩 transition
     bias，从而恢复 block 内的顺序依赖。
   - confidence head 用于估计接受概率，并辅助决定本轮实际送去验证的长度。

主模型验证 draft block 后，已接受 token 成为新的 target-model history，下一轮 DSpark
再基于更新后的 target hidden 和 rolling window 产生新 block。

这意味着 parallel backbone 和 Markov head 解决的是两个不同问题：

- backbone 负责一次并行构造多个位置的强 base distribution；
- Markov head 负责以很低的串行成本注入已采样 draft token 之间的依赖。

## 4. 128 rolling window 到底是什么

DeepSeek-V4-Pro-DSpark 的 production 配置包含三个 MoE+mHC draft layers、128-token
sliding-window attention、最大 draft block size 5 和 Markov head。

128 window 保存的是每个 DSpark attention layer 的近期 context KV：

```text
target model selected hidden states
            │
            ▼
  DSpark main projection / norm
            │
            ▼
 layer-specific KV projection + norm + RoPE
            │
            ▼
 per-layer circular context KV window, capacity = 128
```

在一轮 draft forward 中，draft query 可以关注：

- rolling window 中最多 128 个有效 context KV；
- 当前正在并行生成的 draft block 的临时 KV。

当前 block 的临时 KV 只服务本轮 attention，不直接作为持久 rolling state 保存。只有
token 被 target model 验证、并产生相应 target hidden 后，才以 target-derived KV 的
形式进入持久 window。

### 4.1 128 不代表什么

它不是：

- 一次最多生成 128 个 draft token；
- Markov head 保存的 128-token token history；
- confidence head 的历史状态；
- 主模型 KV cache 的替代品；
- prompt 或模型能够使用的最大上下文长度。

`128` 只是 drafter attention 的局部高分辨率 context 容量。主模型本身仍使用完整的
target KV cache；DSpark 当前轮输入的 target hidden 也已经编码了更长的 prompt/history。

## 5. rolling window 的生成是否依赖 DSpark head

如果“DSpark head”指 Markov head、confidence head 或最终 LM projection，答案是：
**不依赖**。

window 的生成依赖：

- 主模型指定层输出的 target hidden states；
- DSpark backbone 中用于把这些 hidden states 映射为各 draft layer context KV 的参数。

它不依赖：

- draft token sampling；
- Markov transition；
- confidence prediction；
- 本轮 draft block 被接受多少。

因此 prefill 期间只要已经获得需要的主模型 hidden states，就可以独立构造 prompt 尾部
的 128-entry window。没有必要为了构造这个 buffer 而先运行完整的第一轮 DSpark
drafter。

## 6. rolling window 为什么仍然有价值

当前 target hidden 是一条压缩后的全局信号；128-entry window 则保留最近 token 的
逐位置、逐 draft layer KV。两者是互补关系：

| 信号 | 主要作用 |
|---|---|
| 当前 target hidden | 向 drafter 提供已编码完整历史的全局条件 |
| 128 context window | 提供最近 128 个 target positions 的细粒度局部注意力材料 |
| 当前 draft-block KV | 允许 parallel backbone 内的 block-level interaction |
| Markov head | 注入基于已采样 draft token 的轻量顺序依赖 |

因此，丢失 prompt seed 的理论损失不是“drafter 完全看不到 prompt”，而是：

> 第一阶段缺少 prompt 尾部 128 个位置的细粒度 context KV，只剩当前 target hidden
> 提供的压缩全局条件，并随着生成继续逐步用新产生的 target-derived KV 重建 window。

这是后续实验中 seed-off 损失很小的一个关键解释。

## 7. disaggregated serving 中应当在哪里生成和使用

语义上最完整的流程是：

```text
Prefill node
  target prefill
      │
      ├─ main-model KV transfer ─────────────────────┐
      │                                              │
      └─ build last-128 DSpark context KV + ctx_len ─┤
                                                     ▼
Decode node
  install target KV + DSpark seed
      │
      └─ run the first DSpark draft round
```

推荐由 prefill 端：

1. 在 target prefill 中捕获 DSpark 需要的 selected hidden states；
2. 构造并填充 rolling window；
3. 同 window 一起传递有效逻辑长度或等价 cursor；
4. decode 端安装 seed 后，再执行第一轮 draft。

不建议仅为了 seed 提前在 prefill 端执行第一轮完整 drafter，因为：

- drafter 的输出还需要在 decode scheduling 和 sampling 状态下被消费；
- 会引入 proposal、RNG、sampling params 和 request lifecycle 的额外交接；
- rolling state 本身并不依赖该完整 drafter forward。

但这是一项质量/性能优化，而不是正确生成 token 的必要条件。decode 端从空 window
启动，只要有效长度和 mask 正确，仍然是一个定义良好的降级路径。

---

## Part II：TRT-LLM `seed=on` 实现审查

## 8. 当前实现的数据流

TRT-LLM 中观察到的数据流如下：

```text
Context/prefill worker

captured target hiddens
  -> _seed_context_windows()
  -> draft_model.write_context_windows()
  -> per-slot window + ctx_len
  -> take_export_seed()
  -> BF16 auxiliary payload + int64 ctx_len
                        │
                        │ disaggregated transfer
                        ▼
Generation/decode worker

stash_pending_seed()
  -> prepare()
  -> assign request slot
  -> _apply_pending_seed()
  -> first forward_batched()
  -> DSpark attention reads transferred window
```

### 8.1 Prefill 端构造

`tensorrt_llm/_torch/speculative/dspark.py` 的
`_seed_context_windows()`：

- 使用捕获到的 target hidden states；
- 将 `ctx_len` 设置为最后一个 prompt position 加一；
- 使用 token 的实际 absolute position 执行 context-window 写入；
- 调用 `draft_model.write_context_windows()` 生成各层 KV；
- 导出 window 和 `ctx_len`。

`tensorrt_llm/_torch/models/modeling_dspark.py` 的
`write_context_windows()` 完成：

- selected hidden 拼接/映射；
- stage-specific projection；
- KV projection 和 normalization；
- RoPE；
- 写入 circular window。

这与 Hugging Face reference 的状态语义一致：持久 window 存的是
target-hidden-derived、layer-specific 的 context KV，而不是 draft block 的临时 KV。

### 8.2 传输内容与顺序

auxiliary payload 包含：

- BF16 rolling-window tensor；
- int64 `ctx_len`。

以三层、128 window、head dimension 512 计算，每个 request 的 window payload 约为：

```text
3 × 128 × 512 × 2 bytes = 393,216 bytes = 384 KiB
```

另加很小的长度元数据和传输管理开销。

传输实现同时处理 context-first 和 generation-first 注册顺序；decode 端在 seed 已到达时
暂存 pending seed，待 request 获得本地 slot 后应用。这避免了把远端 slot 编号错误地
当成本地 slot 使用。

### 8.3 Decode 端应用时机

`DSparkAlgorithm.prepare()` 的关键顺序为：

1. 为 request 分配/解析本地 slot；
2. 将 pending seed 拷贝到该 slot；
3. 建立 request-to-slot 映射；
4. 随后进入第一次 `forward_batched()`。

因此 seed 不是在第一轮 draft 后才补上，而是在第一轮 attention 读取 cache 前已经安装。
这一点符合预期。

### 8.4 Attention 是否真的消费 seed

`tensorrt_llm/_torch/models/dspark/attention.py` 中：

- 当前 target hidden 先生成当前 position 的 `main_kv`；
- `main_kv` 写入持久 window；
- attention gather 当前 slot 的 circular rows；
- 根据 `start_pos`/有效长度排除未填充 entries；
- 将有效 context KV 和当前 draft-block KV 合并后计算 attention。

所以 transfer 过来的 tensor 并非仅被保存但不使用；它位于 first-draft attention 的实际
读取路径上。

## 9. seed-off 的真实冷启动行为

这是解释实验结果时最容易误判的部分。

新 slot 分配时，TRT-LLM 会：

- 清零 window；
- 设置 `_ctx_len = 0`。

attention 构造 context top-k/mask 时，会用当前逻辑长度排除尚未填充的 slots。因此
seed-off 并不是：

```text
把 128 个全零 KV 当成 128 个有效 prompt tokens 去 attention
```

而是：

```text
初始没有有效的历史 context entries；
仅使用当前 target-derived KV、当前 draft block；
以后只暴露已经被真实 target hidden 回填的 window entries。
```

这个设计使 seed-off 成为稳定的空历史启动，而不是被大批无效零值污染的异常状态。

## 10. 生成期间如何自然回填

每轮 target verification 后，TRT-LLM 不只写入一个位置：

- 当前 round 的 target hidden 会写入；
- 本轮中间已接受 token 对应的 captured target hiddens 也会通过
  `write_context_windows_batched()` 回填；
- 逻辑长度按本轮已接受数量推进。

因此恢复速度约与每轮实际接受 token 数量成正比。若 AL 接近 4，window 大约经过
30 多轮即可包含 128 个新生成 positions，而不是必须经过 128 轮。

需要区分两个说法：

- “第二轮开始可以利用 main model 信息”——正确，第一轮验证后即可开始回填；
- “第二轮就自然填满完整 128 window”——不正确，第二轮只有刚生成并验证的少量
  positions；填满仍需累计约 128 个 accepted target positions。

## 11. `seed=on` 正确性判断

### 11.1 没有发现的错误

本次静态审查没有发现以下问题：

- prefill 端错误地依赖 Markov/confidence head 才能构造 seed；
- 传输了 draft-block 临时 KV，而不是 target-derived context KV；
- 没有传输逻辑长度，导致空槽被当成有效槽；
- 使用 prefill worker 的 slot id 直接寻址 decode worker；
- seed 在第一轮 drafter 结束后才应用；
- attention forward 完全绕过传入 seed；
- context-first/generation-first 的 arrival ordering 丢失 seed。

硬件验证记录还显示，接收侧观察到了非零 window norm、正确的 `ctx_len` 和 byte-exact
payload，这进一步支持 transport 和 application 路径正常。

### 11.2 仍需关注的问题

| 优先级 | 问题 | 影响 |
|---|---|---|
| 高 | 当前代码未消费 `TRTLLM_DSPARK_DISABLE_SEED` | 当前 checkout 无法按现有脚本真正复现 seed-off |
| 高 | 测试未覆盖 first-draft attention/logits 差分 | 能证明 tensor 搬运正确，但不能直接证明 seed 改变模型计算结果 |
| 中 | 每个 prefill chunk 后导出/CPU clone 整个 seed | 可能产生不必要的 D2H 和同步成本，尤其对长 prompt |
| 中 | 启用该 shape 后所有相关 request 都保留 aux transfer 预算 | 每 request 约 384 KiB；收益很小时需重新评估带宽和并发代价 |
| 低 | 旧调查文档中“验证完成”和早期“待验证”章节并存 | 容易把历史计划误读为当前状态，建议标记时间线或归档旧章节 |

### 11.3 单元测试覆盖评价

`tests/unittest/_torch/speculative/test_dspark_disagg_seed.py` 当前主要覆盖：

- payload shape 和 dtype；
- round-trip/copy；
- auxiliary buffer 行为；
- seed 的存取和应用。

尚缺少模型级断言：

```text
同一 prompt、同一 current target hidden、同一 RNG
  seed=on first draft logits
vs
  seed=off first draft logits
```

当前环境未安装可用的 `pytest` 模块，因此本 review 未重新执行该测试文件；静态审查不受
此限制，但测试通过状态需以原硬件/开发环境记录为准。

---

## Part III：为什么 aggregated / seed-on / seed-off 的 AL 差别不大

## 12. 现有实验现象

匹配方法下，N=16、三个数据集、OSL 8/16/32/64/128/256 的平均 AL 如下：

| Dataset | Mode | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---:|---:|---:|---:|---:|---:|
| GSM8K | seed-off | 2.150 | 2.663 | 3.214 | 3.856 | 4.212 | 4.239 |
| GSM8K | seed-on | 2.350 | 2.730 | 3.377 | 3.854 | 4.229 | 4.116 |
| GSM8K | aggregated | 2.175 | 2.919 | 3.273 | 3.847 | 4.209 | 4.468 |
| HumanEval | seed-off | 2.050 | 2.695 | 3.541 | 4.140 | 4.454 | 4.461 |
| HumanEval | seed-on | 1.867 | 2.729 | 3.835 | 4.266 | 4.534 | 4.366 |
| HumanEval | aggregated | 1.908 | 2.781 | 3.694 | 4.324 | 4.466 | 4.558 |
| MATH-500 | seed-off | 1.835 | 2.702 | 3.391 | 3.631 | 4.208 | 4.489 |
| MATH-500 | seed-on | 1.930 | 2.687 | 3.432 | 3.883 | 4.219 | 4.397 |
| MATH-500 | aggregated | 1.888 | 2.716 | 3.504 | 3.811 | 4.195 | 4.414 |

三种模式整体重叠，没有稳定的 seed-on 优势；局部正负波动也不呈单调趋势。

## 13. 最可能的模型与实现原因

### 13.1 空 window 被正确 mask，seed-off 损失有限

如第 9 节所述，seed-off 只是少了历史 context entries，不会 attention 到 128 个伪造
的零 token。这会显著缩小 seed-on/off 的理论差距。

### 13.2 每轮仍有包含完整 prompt 信息的 target hidden

DSpark 的当前 `main_hidden` 来自主模型选定层，并经过主模型对完整 prompt/history 的
处理。即使 prompt rolling window 没有传输，drafter 仍不是无条件生成。

换言之，seed-off 丢失的是最近 128 个 prompt positions 的显式逐位置 KV，而不是丢失
全部 prompt semantics。若 checkpoint 主要依赖当前 target hidden 和 Markov transition，
AL 变化自然可能很小。

### 13.3 window 会按 accepted tokens 快速回暖

TRT-LLM 会回填每轮中间接受 positions。若每轮接受约 4 个 token：

- 约 8 轮后已有约 32 个 generated-context entries；
- 约 16 轮后已有约 64 个；
- 约 32 轮后 prompt seed 基本被生成 suffix 替换。

所以 seed 的潜在影响主要集中在输出开头；在较长 OSL 的 request-level 平均 AL 中会被
快速稀释。

### 13.4 position origin 不同未必造成持续差异

seed-on 使用 prompt 的 absolute positions；seed-off 的本地 context 从零长度开始推进。
对以 RoPE 相对相位为主要作用方式的局部 attention，当 window 最终只包含生成 suffix
时，对所有局部 positions 做统一平移通常近似保持相对位置信息。

这不能证明两条路径严格等价，但解释了为什么冷启动 position origin 的差异未必造成
长期 AL 损失。

### 13.5 window 可能不是该 checkpoint 的主导质量来源

DSpark 还有：

- 当前 target hidden；
- 三层 parallel backbone；
- 当前 5-token block 内 attention；
- Markov transition bias；
- confidence-driven scheduling。

128 context window 可能只在需要精细局部复现、格式或拷贝的 token 上提供额外收益。
GSM8K、HumanEval 和 MATH-500 的平均 AL 未必对此足够敏感。

## 14. 为什么短 OSL 的 AL 会普遍偏低

TRT-LLM 当前 request-level 指标近似为：

```text
AL = generated_tokens / decoding_iterations
```

短请求的最后一轮通常只需要 speculative block 的一部分，超出 OSL 的可用 accepted
tokens 会被截断，但该轮仍完整计入 `decoding_iterations`。例如一个 block 本可贡献
4 个 token，但 request 只剩 1 个输出 token，则统计中该轮贡献为 1。

因此三种模式的 AL 都随着 OSL 增长，并不必然表示“模型经过 128 token 才变暖”；其中
相当一部分是短请求的末轮截断和整轮计数效应。

这也说明 request-level AL 不是观察“最初 128 个输出 token seed 是否有用”的最佳指标。

## 15. 统计与实验开关问题

### 15.1 N=16 难以分辨小效应

当前差异量级通常只有 0.0x～0.2x，而且不同数据集/OSL 方向不一致。在没有 paired
confidence interval 的情况下，无法区分小的真实收益和 request composition / sampling
噪声。

### 15.2 当前 seed-off 开关不可复现

外部运行脚本在 `SEED=off` 时设置：

```bash
TRTLLM_DSPARK_DISABLE_SEED=1
```

但当前仓库搜索不到读取该变量的代码。调查文档记录了验证期间曾存在临时 kill-switch
patch，之后又被回退。

因此应区分：

- **历史数据**：如果是在临时 patch 生效期间采集，可能是有效的 true off；
- **当前重跑**：如果只设置该环境变量而不恢复消费者，on/off 实际都传 seed。

这是可复现性问题，而不是已经证明 seed-on 实现错误。报告结论不应混淆二者。

---

## 16. 建议的最终验证

## 16.1 首先验证模型计算确实依赖 seed

对同一个 request 固定：

- prompt；
- target KV 和 current target hidden；
- sampling parameters；
- RNG seed；
- request slot。

只改变 rolling state，比较第一次 draft forward：

| Case | Window | `ctx_len` | 目的 |
|---|---|---:|---|
| A | 真实 prefill seed | prompt length | 正常 seed-on |
| B | 全零 | 0 | 真实 cold seed-off |
| C | shuffle/randomized seed | prompt length | 确认 attention 对有效 window 内容敏感 |

记录：

- first attention input/cache checksum；
- attention output 的 max/mean absolute difference；
- first-draft logits difference；
- proposal tokens；
- per-position verification acceptance。

判读：

- A/B logits 明显不同但平均 AL 相同：实现生效，模型/benchmark 对 seed 不敏感；
- A/B logits bitwise 相同，而 C 也相同：高度怀疑 cache 路径被 bypass 或 mask 错误；
- A/B 很接近但 C 明显不同：路径有效，但真实 prompt seed 的边际信息很小；
- A/B 只在最初若干轮不同：符合 rolling state 快速回暖的预期。

## 16.2 恢复明确、可测试的 seed-off 控制

不要只依赖未接线的环境变量。更稳妥的验证方式是：

- 在测试或 benchmark 层显式跳过 `stash_pending_seed()`；或
- 提供一个有日志、有单元测试的临时 debug control；
- decode 侧打印/断言首轮 slot 的 `ctx_len`：
  - seed-on 应等于 prompt logical length；
  - seed-off 应为 0。

若不打算把开关作为正式 API，应只用于验证并在实验完成后移除。

## 16.3 使用按输出位置的 paired 指标

比单个 request-level AL 更敏感的统计方式是：

- 对相同 prompts 做 paired on/off；
- 记录输出位置 0～127 每个 speculative cycle 的 accepted length；
- 绘制或汇总：
  - position 0～7；
  - 8～31；
  - 32～63；
  - 64～127；
  - 128 以后；
- 单独排除或标记 request 最后一轮的 OSL truncation；
- 扩大样本量并报告 paired bootstrap confidence interval。

如果 seed 的作用确实集中在 prompt-to-generation 边界，这类指标比 OSL=8 的总体 AL
更容易检测。

## 16.4 在确认无质量收益后评估传输成本

若更严格实验仍显示收益接近零，需要将约 384 KiB/request 的额外传输、buffer budget、
prefill 端 D2H clone 和调度复杂度纳入决策：

- 对高并发短请求，固定 per-request payload 可能比生成端的短暂 AL 收益更昂贵；
- 对长输出，seed 很快被生成 suffix 替换，收益更容易被摊薄；
- 对强局部依赖 workload，仍可能有不同结论，不能仅由三个当前数据集外推到全部流量。

同时可以优化为仅在 prefill 最终 chunk 导出一次 seed，避免每个 chunk 重复 clone 整个
window。

## 17. 最终评价

从机制和代码路径看，TRT-LLM 当前 `seed=on` 实现与 DSpark reference 的核心语义一致：

- prefill 端构造 target-derived per-layer context KV；
- 连同逻辑长度传给 decode；
- decode 在第一次 drafter forward 前安装；
- attention 使用逻辑长度屏蔽无效槽，并读取有效 seed；
- 后续用 target-verified hidden states 持续维护同一 window。

因此，现有 AL 无明显差别更可能说明 rolling seed 在当前 checkpoint、数据集和统计方法下
的边际收益很小，而不是明显的实现故障。特别是 seed-off 的 graceful masking、每轮仍有
完整上下文编码的 current target hidden，以及 accepted-token 批量回填，共同使 cold
start 的代价远小于“缺失 128-token 上下文”的字面直觉。

不过，在关闭此项工作前仍应完成两个证据链：

1. 恢复并验证真正的 seed-off control，避免当前脚本 on/off 实际同路；
2. 证明 first-draft attention/logits 在真实 seed、cold zero seed 和 randomized seed
   之间按预期产生差异。

完成这两项后，若 paired per-position acceptance 仍无稳定提升，就可以较有把握地得出：

> seed handoff 的实现是正确的，但对已测试场景的质量收益不足以抵消其传输和系统复杂度。
