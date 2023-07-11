# AttentionRollout ReImplementation
Adapt original reimplementation of Attention Rollout by https://github.com/jacobgil/vit-explain.
Original paper for Attention Rollout: https://arxiv.org/pdf/2005.00928.pdf 

**Motivation: ***Isolate the impact of each following topic*** on how the model pay attentions, its accuracy and GPUtime (Flop and GFlop):**

- Representation Learning ViT:
     - [ ] Self-supervised Vision Transformer 2021: https://github.com/Sara-Ahmed/SiT
           
- Different optimizers:
     - [ ] SAM: https://github.com/davda54/sam
     - [ ] Adaptive policy SAM: https://github.com/SamsungLabs/ASAM
     - [ ] Sparse SAM: https://github.com/Mi-Peng/Sparse-Sharpness-Aware-Minimization

- Different Attention-layer in ViT:
     - [ ] Hydra Attention (num_heads=embed_dim=768) to reduce computation while keeping competitive accuracys: https://arxiv.org/abs/2209.07484, (ReImp): https://github.com/robflynnyh/hydra-linear-attention 
     - [ ] (Dense+Sparse) Attention = Attention Retractable Transformer for accurate image restoration: https://github.com/gladzhang/ART - The proposed model is innovative. It combines dense and sparse attention modules. The proposed sparse attention can allow token interactions in sparse image regions and thus enlarge the receptive field of the module. The combination of sparse and dense modules **allows for global and local interactions while being tractable**.
          

- U-Net backbone for Diffusion Model:
     - [ ] Art style diffusion: https://github.com/ChenDarYen/ArtFusion
     - [ ] Prompt-to-Prompt Image Editing with Cross-Attention Control: https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/2725. *Learning how to use to general stable-diffusion is a tough learning curve **but worth it in long term**

- ViT backbone for Diffusion Model:
     - [ ] Fast Training of Diffusion Models with Masked Transformers: https://github.com/Anima-Lab/MaskDiT
     - [ ] **VDT: An Empirical Study for Video Diffusion with Transformers**: https://github.com/RERV/VDT
     - [ ] Masked Diffusion Transformer is a Strong Image Synthesizer: https://github.com/sail-sg/MDT
     - [ ] Paper to read as baseline: Exploring Transformer Backbones for Image Diffusion Models


## Other tools to use other than Attention Rollout ##
- [ ] Attention Rollout
- [ ] Gradient-based Attention Rollout
- [ ] ????
