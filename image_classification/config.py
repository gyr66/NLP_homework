class Config:
    data_path = "./data/large"
    model = "resnet"
    batch_size = 512
    lr = 0.0001
    clip = 1
    epochs = 50
    saved = True
    multi_gpu = False
    only_test = True
    gpus = [1, 3, 4, 5, 6, 7]
