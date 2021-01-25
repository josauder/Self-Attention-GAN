import torch
import torchvision.datasets as dsets
from torchvision import transforms

def AddComplexZeros(x):
  """ Add complex channel to images filled with zeros to make things compatible with fft2/ifft2"""
  x = x.repeat(2, 1, 1)
  x[1] = 0
  return x


class Data_Loader():
    def __init__(self, train, dataset, image_path, image_size, batch_size, shuf=True):
        self.dataset = dataset
        self.path = image_path
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf
        self.train = train

    def transform(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def load_lsun(self, classes='church_outdoor_train'):
        transforms = self.transform(True, True, True, False)
        dataset = dsets.LSUN(self.path, classes=[classes], transform=transforms)
        return dataset

    def load_celeb(self):
        transforms = self.transform(True, True, True, True)
        dataset = dsets.ImageFolder(self.path+'/CelebA', transform=transforms)
        return dataset

    def load_cifar100bw(self):
        transform = transforms.Compose([transforms.Grayscale(),
                                   transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ToTensor()])
        dataset = dsets.CIFAR100(self.path+'/Cifar100bw', train=True, transform=transform, target_transform=None,
                                          download=True)

        return dataset

    def load_cifar100bw_test(self):
        transform = transforms.Compose([transforms.Grayscale(),
                                   transforms.ToTensor()])
        dataset = dsets.CIFAR100(self.path+'/Cifar100bw_test', train=False, transform=transform, target_transform=None,
                                          download=True)

        return dataset

    def loader(self):
        if self.dataset == 'lsun':
            dataset = self.load_lsun()
        elif self.dataset == 'celeb':
            dataset = self.load_celeb()
        elif self.dataset == 'cifar100bw':
            dataset = self.load_cifar100bw()
        elif self.dataset == 'cifar100bw_test':
            dataset = self.load_cifar100bw_test()

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=self.batch,
                                              shuffle=self.shuf,
                                              num_workers=2,
                                              drop_last=True)
        return loader

