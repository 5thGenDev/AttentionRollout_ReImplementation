# AttentionRollout ReImplementation
- Original paper: https://arxiv.org/pdf/2005.00928.pdf. 
- Motivation: Isolate the impact of each attention block in ViT for image classification task in term of its accuracy

     
## Different Attention-layer in ViT:
- [x] Hydra Attention (num_heads=embed_dim=768) to reduce computation while keeping competitive accuracys: https://arxiv.org/abs/2209.07484
- [ ] (Dense+Sparse) Attention = Attention Retractable Transformer for accurate image restoration: https://github.com/gladzhang/ART 
          
## Other tools to use other than Attention Rollout
- [ ] Attention Rollout
- [ ] Gradient-based Attention Rollout
- [ ] ????
