from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DataParallel
from model import ET_Net
from args import ARGS
import time
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from utils.get_dataset import get_dataset
from utils.lovasz_losses import lovasz_softmax
import os

from matplotlib import pyplot as plt

class TrainValProcess():
    def __init__(self):
        self.net = ET_Net()
        if (ARGS['weight']):
            self.net.load_state_dict(torch.load(ARGS['weight']))
        else:
            self.net.load_encoder_weight()
        if (ARGS['gpu']):
            self.net = DataParallel(module=self.net.cuda())
        
        self.train_dataset = get_dataset(dataset_name=ARGS['dataset'], part='train')
        self.val_dataset = get_dataset(dataset_name=ARGS['dataset'], part='val')

        self.optimizer = Adam(self.net.parameters(), lr=ARGS['lr'])
        # Use / to get an approximate result, // to get an accurate result
        total_iters = len(self.train_dataset) // ARGS['batch_size'] * ARGS['num_epochs']
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda iter: (1 - iter / total_iters) ** ARGS['scheduler_power'])
        self.writer = SummaryWriter()

    def train(self, epoch):

        start = time.time()
        self.net.train()
        train_dataloader = DataLoader(self.train_dataset, batch_size=ARGS['batch_size'], shuffle=False)
        epoch_loss = 0.
        for batch_index, items in enumerate(train_dataloader):
            images, labels, edges = items['image'], items['label'], items['edge']
            images = images.float()
            labels = labels.long()
            edges = edges.long()

            if ARGS['gpu']:
                labels = labels.cuda()
                images = images.cuda()
                edges = edges.cuda()

            self.optimizer.zero_grad()
            outputs_edge, outputs = self.net(images)
            # print('output edge min:', outputs_edge[0, 1].min(), ' max: ', outputs_edge[0, 1].max())
            # plt.imshow(outputs_edge[0, 1].detach().cpu().numpy() * 255, cmap='gray')
            # plt.show()
            loss_edge = lovasz_softmax(outputs_edge, edges) # Lovasz-Softmax loss
            loss_seg = lovasz_softmax(outputs, labels) # 
            loss = ARGS['combine_alpha'] * loss_seg + (1 - ARGS['combine_alpha']) * loss_edge
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            n_iter = (epoch - 1) * len(train_dataloader) + batch_index + 1
            
            pred = torch.max(outputs, dim=1)[1]
            iou = torch.sum(pred & labels) / (torch.sum(pred | labels) + 1e-6)

            # print('edge min:', edges.min(), ' max: ', edges.max())
            # print('output edge min:', outputs_edge.min(), ' max: ', outputs_edge.max())

            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tL_edge: {:0.4f}\tL_seg: {:0.4f}\tL_all: {:0.4f}\tIoU: {:0.4f}\tLR: {:0.4f}'.format(
                loss_edge.item(),
                loss_seg.item(),
                loss.item(),
                iou.item(),
                self.optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * ARGS['batch_size'],
                total_samples=len(train_dataloader.dataset)
            ))

            epoch_loss += loss.item()

            # update training loss for each iteration
            # self.writer.add_scalar('Train/loss', loss.item(), n_iter)

        for name, param in self.net.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            self.writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

        epoch_loss /= len(train_dataloader)
        self.writer.add_scalar('Train/loss', epoch_loss, epoch)
        finish = time.time()

        print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

    def validate(self, epoch):

        start = time.time()
        self.net.eval()
        val_batch_size = min(ARGS['batch_size'], len(self.val_dataset))
        val_dataloader = DataLoader(self.val_dataset, batch_size=val_batch_size)
        epoch_loss = 0.
        for batch_index, items in enumerate(val_dataloader):
            images, labels, edges = items['image'], items['label'], items['edge']
            # print('label min:', labels[0].min(), ' max: ', labels[0].max())
            # print('edge min:', labels[0].min(), ' max: ', labels[0].max())

            if ARGS['gpu']:
                labels = labels.cuda()
                images = images.cuda()
                edges = edges.cuda()
            
            print('image shape:', images.size())

            with torch.no_grad():
                outputs_edge, outputs = self.net(images)
                loss_edge = lovasz_softmax(outputs_edge, edges) # Lovasz-Softmax loss
                loss_seg = lovasz_softmax(outputs, labels) # 
                loss = ARGS['combine_alpha'] * loss_seg + (1 - ARGS['combine_alpha']) * loss_edge
            
            pred = torch.max(outputs, dim=1)[1]
            iou = torch.sum(pred & labels) / (torch.sum(pred | labels) + 1e-6)

            print('Validating Epoch: {epoch} [{val_samples}/{total_samples}]\tLoss: {:0.4f}\tIoU: {:0.4f}'.format(
                loss.item(),
                iou.item(),
                epoch=epoch,
                val_samples=batch_index * val_batch_size,
                total_samples=len(val_dataloader.dataset)
            ))

            epoch_loss += loss

            # update training loss for each iteration
            # self.writer.add_scalar('Train/loss', loss.item(), n_iter)
        
        epoch_loss /= len(val_dataloader)
        self.writer.add_scalar('Val/loss', epoch_loss, epoch)

        finish = time.time()

        print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    
    def train_val(self):
        print('Begin training and validating:')
        for epoch in range(ARGS['num_epochs']):
            self.train(epoch)
            self.validate(epoch)
            self.net.state_dict()
            print(f'Finish training and validating epoch #{epoch+1}')
            if (epoch + 1) % ARGS['epoch_save'] == 0:
                os.makedirs(ARGS['weight_save_folder'], exist_ok=True)
                torch.save(self.net.state_dict(), os.path.join(ARGS['weight_save_folder'], f'epoch_{epoch+1}.pth'))
                print(f'Model saved for epoch #{epoch+1}.')
        print('Finish training and validating.')

if __name__ == "__main__":
    tv = TrainValProcess()
    tv.train_val()