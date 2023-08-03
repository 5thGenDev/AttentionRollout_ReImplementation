import torch
import torch.backends.cudnn as cudnn

def main():
    # Warn user to use GPUs
    if not args.use_avai_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    if use_gpu:
        print(f"Currently using GPU {args.gpu_devices}")
        cudnn.benchmark = True
    else:
        warnings.warn("Currently using CPU, however, GPU is highly recommended")
    
    #  Create Model
    device = "cuda"
    model = vit_small(16,img_size=224, num_classes=10)
    model = model.to(device)
    
    # Load the pretrained weights
    state = torch.load("~dir_to_file/SiT_Small_ImageNet.pth")
    state = {k.replace("module.backbone.",""):v for k,v in state['teacher'].items()} # extract the model weights!
    print("Loading weights: ", model.load_state_dict(state,strict=False))


if __name__ == "__main__":
  main()
