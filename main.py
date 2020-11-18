from train import TrainValProcess

if __name__ == "__main__":
    state = 'train' # train or test
    if (state == 'train'):
        tv = TrainValProcess()
        tv.train_val()