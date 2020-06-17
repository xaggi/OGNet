
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch

def load_data(opt):
    dataset = dset.ImageFolder(root=opt.data_path,  # opt.data_path
                               transform=transforms.Compose([
                                   transforms.Grayscale(),
                                   transforms.Resize((45, 45)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5], [0.5]),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=True,
                                             num_workers=opt.n_threads,
                                             drop_last=opt.drop_last)
    return dataloader