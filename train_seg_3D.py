from parameters import batch_size, learning_rate, learning_rate_decay, Epoch

import torch
import datetime
import logging
import pandas as pd
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# Dataset
def get_dl(test=False):
    from dataset_prepare.dataset import Dataset
    from torch.utils.data import DataLoader

    train_ds = Dataset(shape='3D', test=test)
    train_dl = DataLoader(train_ds, batch_size, shuffle=False)

    return train_dl


# net
def get_model():
    from net.Unet_3D import Unet

    model = Unet(in_ch=1, out_ch=1).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(optimizer, learning_rate_decay)

    return model, optimizer, lr_decay


# Loss
def get_loss_fuction():
    from loss.Dice import DiceLoss
    loss_function = DiceLoss()

    return loss_function


# test
def test(model):
    test_dl = get_dl(test=True)
    model.eval()

    ls = []
    for step, (img, mask, _, _) in enumerate(test_dl):
        if torch.cuda.is_available():
            img, mask = img.cuda(), mask.cuda()

        pred = model(img)
        loss = loss_function(pred, mask)

        ls.append(loss.item())

    return sum(ls) / len(ls)


# train
def train():
    losses = []
    for epoch in range(Epoch):
        loss_all = 0
        model.train()
        for step, (img, mask, _) in enumerate(train_dl):
            if torch.cuda.is_available():
                img, mask = img.cuda(), mask.cuda()

            pred = model(img)
            loss = loss_function(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_decay.step()

            loss_all += loss.item()
            logging.info(
                'time:%s | step:%d/%d/%d | loss=%.6f' % (datetime.datetime.now(), step, epoch, Epoch, loss.item()))

        train_loss = loss_all / len(train_dl)

        # test
        if epoch % 10 == 0 and epoch != 0:
            test_loss = test(model)
            print('epoch:%d, test_loss:%f' % (epoch, test_loss))
            losses.append([train_loss, test_loss])
        else:
            losses.append([train_loss, '-'])

        # save
        if epoch % 5 == 0 and epoch != 0:
            torch.save(model.state_dict(), './saved_models/seg_3D/net{}-{:.3f}.pth'.format(epoch, train_loss))

    return losses


def save_loss(losses):
    # save train and eval loss
    df = pd.DataFrame(data=losses)
    df.columns = ['loss', 'test']
    df.to_csv('./out/loss/seg_3D:%s.csv' % datetime.datetime.now())


if __name__ == '__main__':
    train_dl = get_dl()
    model, optimizer, lr_decay = get_model()
    loss_function = get_loss_fuction()

    losses = train()

    save_loss(losses)
