from torch import nn

class Hook:
    def __init__(self,model: nn.Module, module='attn.attn_drop',mode="output") -> None:
        """set a hook to get the intermedia results

        Args:
            model (nn.Module): ViTs
            module (str, optional): Name of module. 'attn.attn_drop' for attn matrix; 'block' for outputs of blocks. Defaults to 'attn.attn_drop'.
        """
        assert mode in ["input", "output"]
        self.mode = mode 
        self.model = model
        self.module = module
    
    
    def register_hook(self):
        for name, m in self.model.named_modules():
            if name.endswith(self.module):
                yield m.register_forward_hook(self._hook)
    
    def _hook(self,m,input,output):
        if self.mode == "output":
            self.outputs.append(output)
        else:
            self.outputs.append(input)
        
    def __call__(self, *args,**kwargs):
        self.outputs=[]
        hook_handlers = list(self.register_hook())
        out=self.model(*args,**kwargs)
        for h in hook_handlers: h.remove()
        
        return out,self.outputs
    
