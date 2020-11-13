import torchvision

if __name__ == "__main__":
    a = torchvision.models.resnet50(pretrained=True, progress=True)