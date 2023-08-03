# AttentionRollout ReImplementation
- Original paper: https://arxiv.org/pdf/2005.00928.pdf. 
- Motivation: Visualize each type of attention block and isolate their impact to image classification accuracy for ViT model.

     
## Other Attention in ViT:
- [x] Hydra Attention (num_heads=embed_dim=768):  linear complexity while keeping competitive accuracys: https://arxiv.org/abs/2209.07484
- [ ] Dilated-Self Attention used for LongNet: Also linear complexity. Reimplemented by https://github.com/alexisrozhkov/dilated-self-attention  
          
## Other than Attention Rollout
- [ ] Attention Rollout
- [ ] Gradient-based Attention Rollout
- [ ] ????
