# AttentionRollout ReImplementation
Original paper for Attention Rollout: https://arxiv.org/pdf/2005.00928.pdf 

## Motivation: Isolate the impact of each attention block in ViT for image classification task in term of its accuracy

     
## Different Attention-layer in ViT:
    - [x] Hydra Attention (num_heads=embed_dim=768) to reduce computation while keeping competitive accuracys: https://arxiv.org/abs/2209.07484, (ReImp): https://github.com/robflynnyh/hydra-linear-attention 
    - [ ] (Dense+Sparse) Attention = Attention Retractable Transformer for accurate image restoration: https://github.com/gladzhang/ART - The proposed model is innovative. It combines dense and sparse attention modules. The proposed sparse attention can allow token interactions in sparse image regions and thus enlarge the receptive field of the module. The combination of sparse and dense modules **allows for global and local interactions while being tractable**.
          
## Other tools to use other than Attention Rollout
- [ ] Attention Rollout
- [ ] Gradient-based Attention Rollout
- [ ] ????
