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

    train_ds = Dataset(shape='2D', test=test)
    train_dl = DataLoader(train_ds, batch_size, shuffle=False)

    return train_dl


# net
def get_model():
    from net.pre_trained_models import ResNet18
    model = ResNet18(in_ch=1).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(optimizer, learning_rate_decay)

    return model, optimizer, lr_decay


# Loss
def get_loss_fuction():
    from torch.nn import BCELoss
    loss_function = BCELoss()

    return loss_function


# test
def test(model):
    test_dl = get_dl(test=True)
    model.eval()

    ls = []
    for step, (img, _, label, _) in enumerate(test_dl):
        if torch.cuda.is_available():
            img, label = img.cuda(), label.cuda()

        pred_label = model(img)
        loss = loss_function(pred_label, label)

        ls.append(loss.item())

    return sum(ls) / len(ls)


# train
def train():
    losses = []
    for epoch in range(Epoch):
        loss_all = 0
        model.train()
        for step, (img, _, label) in enumerate(train_dl):
            if torch.cuda.is_available():
                img, label = img.cuda(), label.cuda()

            pred_label = model(img)
            loss = loss_function(pred_label, label)

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
            torch.save(model.state_dict(), './saved_models/cls/net{}-{:.3f}.pth'.format(epoch, train_loss))

    return losses


def save_loss(losses):
    # save train and eval loss
    df = pd.DataFrame(data=losses)
    df.columns = ['loss', 'test']
    df.to_csv('./out/loss/cls:%s.csv' % datetime.datetime.now())


if __name__ == '__main__':
    train_dl = get_dl()
    model, optimizer, lr_decay = get_model()
    loss_function = get_loss_fuction()

    losses = train()

    save_loss(losses)
