# AttentionRollout ReImplementation
- Original paper: https://arxiv.org/pdf/2005.00928.pdf. 
- Motivation: Visualize each type of attention block and isolate their impact to image classification accuracy for ViT model.

     
## Other Attention in ViT:
Note that d_model = embed_dim already where d_model = number of tokens, head_dim = d_model/num_heads
- [x] [Hydra Attention](https://arxiv.org/abs/2209.07484) argues for num_heads = embed_dim to get linear complexity. Replacing 2 Self Attention layers from the back with  Hydra Attention improved accuracy while reduced FLOPs and runtime. Reimplemented by https://github.com/robflynnyh/hydra-linear-attention/blob/main/hydra.py. Unfortunately, Hydra Attention requires a different math to visualise its attention map. However, figure 3 + appendix shown that despite different math, the attention map is pretty much the same
- [ ] Dilated-Self Attention used for LongNet: Also linear complexity. Reimplemented by https://github.com/alexisrozhkov/dilated-self-attention  
          
## Other than Attention Rollout
- [ ] Attention Rollout
- [ ] Gradient-based Attention Rollout
- [ ] ????
