GPT-OSS-120B MoE with EP=8, T=16384, H=2880, K=4, E=128, E_local=16, R=T*K=65536

FORWARD
=======
X [16384,2880]
  ├── router/topk ───────────────► expert_index [16384,4]
  ├── router/topk ───────────────► routing_weights [16384,4]
  └── build expert assignments ──► expert_mask [128,4,16384]
                                     │
                                     └── preprocess + token_pre_all2all
                                           │
                                           └── on each EP rank p:
                                               permute_tokens^(p) [R_p,2880]
                                               cumsum^(p) [16]
                                               fc1_weight^(p) [16,2880,5760]
                                               fc1_bias^(p)   [16,5760]
                                               fc2_weight^(p) [16,2880,2880]
                                               fc2_bias^(p)   [16,2880]
                                                   │
                                                   ├── FC1 grouped GEMM
                                                   │     └── Z^(p) [R_p,5760]
                                                   │
                                                   ├── split/interleave
                                                   │     ├── gate^(p) [R_p,2880]
                                                   │     └── up^(p)   [R_p,2880]
                                                   │
                                                   ├── fused gate op
                                                   │     └── Hmid^(p) [R_p,2880]
                                                   │
                                                   └── FC2 grouped GEMM
                                                         └── Y^(p) [R_p,2880]
                                           │
                                           └── tokens_post_all2all
                                                 └── routed_outputs [16384,4,2880]
                                                        │
                                                        └── weight + reduce over K
                                                              using routing_weights [16384,4]
                                                              └── out [16384,2880]


BACKWARD
========
grad_out [16384,2880]
  ├── grad wrt combine over K
  │     ├── grad_routed [16384,4,2880]
  │     └── grad_routing_weights [16384,4]
  │
  └── reverse tokens_post_all2all
        │
        └── on each EP rank p:
            grad_Y^(p) [R_p,2880]
            saved: Z^(p) [R_p,5760], permute_tokens^(p) [R_p,2880], cumsum^(p) [16]
                │
                ├── FC2 backward
                │     ├── grad_Hmid^(p) [R_p,2880]
                │     ├── grad_fc2_weight^(p) [16,2880,2880]
                │     └── grad_fc2_bias^(p)   [16,2880]
                │
                ├── fused gate backward
                │     └── grad_Z^(p) [R_p,5760]
                │
                └── FC1 backward
                      ├── grad_permute_tokens^(p) [R_p,2880]
                      ├── grad_fc1_weight^(p) [16,2880,5760]
                      └── grad_fc1_bias^(p)   [16,5760]
        │
        └── reverse token_pre_all2all
              └── grad_routed_input [16384,4,2880]
                    │
                    └── reduce over K
                          └── grad_X [16384,2880]


Notes:
- R_p = number of routed rows landing on EP rank p, with sum_p R_p = 65536
- Each rank owns only 16 local experts
- All-to-all happens twice in forward/backward:
  1) send token-expert pairs to owning expert rank
  2) send expert outputs / input grads back to original token order

#============================================================================
VERSION 2
#============================================================================



GPT-OSS-120B MoE on 1x p5en node (8x H200), EP=8
T=16384, H=2880, K=4, E=128, E_local=16, R=T*K=65536

Rank/GPU mapping
----------------
EP rank 0 = GPU0 : experts   0..15
EP rank 1 = GPU1 : experts  16..31
EP rank 2 = GPU2 : experts  32..47
EP rank 3 = GPU3 : experts  48..63
EP rank 4 = GPU4 : experts  64..79
EP rank 5 = GPU5 : experts  80..95
EP rank 6 = GPU6 : experts  96..111
EP rank 7 = GPU7 : experts 112..127


FORWARD
=======
X [16384,2880]
  ├── router/topk ───────────────► expert_index [16384,4]
  ├── router/topk ───────────────► routing_weights [16384,4]
  └── build expert assignments ──► expert_mask [128,4,16384]
                                     │
                                     └── preprocess + token_pre_all2all
                                         (intra-node all-to-all across GPU0..GPU7)
                                           │
                                           └── on each EP rank p = GPU p:
                                               permute_tokens^(p) [R_p,2880]
                                               cumsum^(p) [16]
                                               fc1_weight^(p) [16,2880,5760]
                                               fc1_bias^(p)   [16,5760]
                                               fc2_weight^(p) [16,2880,2880]
                                               fc2_bias^(p)   [16,2880]
                                                   │
                                                   ├── FC1 grouped GEMM
                                                   │     └── Z^(p) [R_p,5760]
                                                   │
                                                   ├── split/interleave
                                                   │     ├── gate^(p) [R_p,2880]
                                                   │     └── up^(p)   [R_p,2880]
                                                   │
                                                   ├── fused gate op
                                                   │     └── Hmid^(p) [R_p,2880]
                                                   │
                                                   └── FC2 grouped GEMM
                                                         └── Y^(p) [R_p,2880]
                                           │
                                           └── tokens_post_all2all
                                               (intra-node all-to-all across GPU0..GPU7)
                                                 └── routed_outputs [16384,4,2880]
                                                        │
                                                        └── weight + reduce over K
                                                              using routing_weights [16384,4]
                                                              └── out [16384,2880]


BACKWARD
========
grad_out [16384,2880]
  ├── grad wrt combine over K
  │     ├── grad_routed [16384,4,2880]
  │     └── grad_routing_weights [16384,4]
  │
  └── reverse tokens_post_all2all
      (intra-node all-to-all across GPU0..GPU7)
        │
        └── on each EP rank p = GPU p:
            grad_Y^(p) [R_p,2880]
            saved: Z^(p) [R_p,5760], permute_tokens^(p) [R_p,2880], cumsum^(p) [16]
                │
                ├── FC2 backward
                │     ├── grad_Hmid^(p) [R_p,2880]
                │     ├── grad_fc2_weight^(p) [16,2880,2880]
                │     └── grad_fc2_bias^(p)   [16,2880]
                │
                ├── fused gate backward
                │     └── grad_Z^(p) [R_p,5760]
                │
                └── FC1 backward
                      ├── grad_permute_tokens^(p) [R_p,2880]
                      ├── grad_fc1_weight^(p) [16,2880,5760]
                      └── grad_fc1_bias^(p)   [16,5760]
        │
        └── reverse token_pre_all2all
            (intra-node all-to-all across GPU0..GPU7)
              └── grad_routed_input [16384,4,2880]
                    │
                    └── reduce over K
                          └── grad_X [16384,2880]


Notes
-----
- R_p = routed rows landing on GPU p, with sum_p R_p = 65536
- Each GPU owns 16 local experts
- All expert traffic stays inside the single 8-GPU node
- All-to-all happens twice in forward/backward:
  1) send token-expert pairs to the GPU that owns the expert
  2) send expert outputs / input grads back to original token order
