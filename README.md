# AttentionRollout ReImplementation
Adapt original reimplementation of Attention Rollout by https://github.com/jacobgil/vit-explain.
Original paper for Attention Rollout: https://arxiv.org/pdf/2005.00928.pdf 

**Motivation: ***Isolate the impact of each following topic*** on how ViT attends in image classification task, its accuracy and GPUtime (Flop and GFlop):**

        
- Different optimizers:
     - [ ] SAM: https://github.com/davda54/sam
     - [ ] Adaptive policy SAM: https://github.com/SamsungLabs/ASAM
     - [ ] Sparse SAM: https://github.com/Mi-Peng/Sparse-Sharpness-Aware-Minimization

- Different Attention-layer in ViT:
     - [x] Hydra Attention (num_heads=embed_dim=768) to reduce computation while keeping competitive accuracys: https://arxiv.org/abs/2209.07484, (ReImp): https://github.com/robflynnyh/hydra-linear-attention 
     - [ ] (Dense+Sparse) Attention = Attention Retractable Transformer for accurate image restoration: https://github.com/gladzhang/ART - The proposed model is innovative. It combines dense and sparse attention modules. The proposed sparse attention can allow token interactions in sparse image regions and thus enlarge the receptive field of the module. The combination of sparse and dense modules **allows for global and local interactions while being tractable**.
          
## Other tools to use other than Attention Rollout ##
- [ ] Attention Rollout
- [ ] Gradient-based Attention Rollout
- [ ] ????
