import torch.nn as nn
import math

class CLSHead(nn.Module):
    def __init__(self, in_dim, bottleneck_dim, nlayers=3, hidden_dim=4096):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            layers.append(nn.BatchNorm1d(bottleneck_dim, affine=False))
            
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return x


class RECHead(nn.Module):
    def __init__(self, in_dim, in_chans=3, patch_size=16):
        super().__init__()

        layers = [nn.Linear(in_dim, in_dim)]
        layers.append(nn.GELU())
        layers.append(nn.Linear(in_dim, in_dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(in_dim, in_dim))
        layers.append(nn.GELU())

        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        
        self.convTrans = nn.ConvTranspose2d(in_dim, in_chans, kernel_size=(patch_size, patch_size), 
                                                stride=(patch_size, patch_size))


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        
        x_rec = x.transpose(1, 2)
        out_sz = tuple( (  int(math.sqrt(x_rec.size()[2]))  ,   int(math.sqrt(x_rec.size()[2])) ) )
        x_rec = self.convTrans(x_rec.unflatten(2, out_sz))
                
                
        return x_rec
