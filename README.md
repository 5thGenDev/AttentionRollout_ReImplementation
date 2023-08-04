# AttentionRollout ReImplementation
- Original paper: https://arxiv.org/pdf/2005.00928.pdf. 
- Motivation: Visualize each type of attention block and isolate their impact to image classification accuracy for ViT model.

## To compare different pretrained models
- ***Pretrain new Attention-ViT for 800 epochs vs finetuning pretrained SiT for 100 epoches***
- Compare the difference of their attention map. My hypothesis is that there isn't much difference from the point of view of attention map (see Hydra Attention vs Self Attention) so we need an alternative visualiser to describe different attention dynamics: 1 candidate would be weight-allocation difference among pixels in each a filter and between the  filters (see 2022U-Net non-isomorphic)
     
## Other Attention in ViT:
Note that d_model = embed_dim already where d_model = number of tokens, head_dim = d_model/num_heads
- [x] [Hydra Attention](https://arxiv.org/abs/2209.07484) argues for num_heads = embed_dim to get linear complexity. Have 2 Hydra Attention-Encoder block at the back improved accuracy while reduced FLOPs and runtime. Reimplemented by https://github.com/robflynnyh/hydra-linear-attention/blob/main/hydra.py. Unfortunately, visualize Hydra Attention needed a different math so we will rely on their (figure 3 + appendix) to discuss different pretrained model
- [ ] Dilated-Self Attention used for LongNet: Also linear complexity. Reimplemented by https://github.com/alexisrozhkov/dilated-self-attention  
          
## Other than Attention Rollout
- [ ] Attention Rollout
- [ ] Gradient-based Attention Rollout
- [ ] ????
