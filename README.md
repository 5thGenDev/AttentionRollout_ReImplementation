# AttentionRollout ReImplementation
- Original paper: https://arxiv.org/pdf/2005.00928.pdf. 
- Motivation: Visualize each type of attention block and isolate their impact to image classification accuracy for ViT model.

     
## Other Attention in ViT:
Note that d_model = embed_dim already where d_model = number of tokens, head_dim = d_model/num_heads
- [x] [Hydra Attention](https://arxiv.org/abs/2209.07484) argues for num_heads = embed_dim to get linear complexity. Have 2 Hydra Attention-Encoder block at the back improved accuracy while reduced FLOPs and runtime. Reimplemented by [robflynnyh](https://github.com/robflynnyh/hydra-linear-attention/blob/main/hydra.py). Unfortunately, visualize Hydra Attention needed a different math so we will rely on their (figure 3 + appendix) to discuss different pretrained model
- [ ] Dilated-Self Attention used for LongNet: Also linear complexity. Reimplemented by https://github.com/alexisrozhkov/dilated-self-attention  
          
## Other than Attention Rollout
- [x] Attention Rollout
- [ ] Gradient-based Attention Rollout
- [ ] ????
