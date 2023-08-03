# AttentionRollout ReImplementation
- Original paper: https://arxiv.org/pdf/2005.00928.pdf. 
- Motivation: Visualize each type of attention block and isolate their impact to image classification accuracy for ViT model.

     
## Other Attention in ViT:
Note that d_model = embed_dim already where d_model = number of tokens, head_dim = d_model/num_heads
- [x] [Hydra Attention](https://arxiv.org/abs/2209.07484) argues for num_heads = embed_dim to get linear complexity while keeping competitive accuracy. Reimplemented by  https://github.com/robflynnyh/hydra-linear-attention/blob/main/hydra.py.
- [ ] Dilated-Self Attention used for LongNet: Also linear complexity. Reimplemented by https://github.com/alexisrozhkov/dilated-self-attention  
          
## Other than Attention Rollout
- [ ] Attention Rollout
- [ ] Gradient-based Attention Rollout
- [ ] ????
