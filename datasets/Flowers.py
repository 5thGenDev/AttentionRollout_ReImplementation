import PIL
from torchvision import transforms

# now load a train set and a validation set
data = datasets.Flowers102("data",transform=build_transform(True,224),download=True, )
train_dl = torch.utils.data.DataLoader(data,64,shuffle=True,num_workers=8,pin_memory=True)

data = datasets.Flowers102("data",False,transform=build_transform(False,224),download=True)
