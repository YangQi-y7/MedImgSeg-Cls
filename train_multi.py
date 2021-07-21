from parameters import batch_size, learning_rate, learning_rate_decay, Epoch, alpha
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
    from net.multi_task_model import MultiTaskModel
    model = MultiTaskModel().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(optimizer, learning_rate_decay)

    return model, optimizer, lr_decay


# Loss
def get_loss_fuction():
    from loss.Dice import DiceLoss
    from torch.nn import BCELoss
    seg_loss_func = DiceLoss()
    cls_loss_func = BCELoss()

    return seg_loss_func, cls_loss_func


# test
def test(model):
    test_dl = get_dl(test=True)
    model.eval()

    loss_all = 0
    seg_loss_all = 0
    cls_loss_all = 0
    for step, (img, mask, label, _) in enumerate(test_dl):
        if torch.cuda.is_available():
            img, mask, label = img.cuda(), mask.cuda(), label.cuda()

        pred_mask, pred_label = model(img)

        seg_loss = seg_loss_func(pred_mask, mask)
        cls_loss = cls_loss_func(pred_label, label)
        loss = seg_loss + (alpha * cls_loss)

        loss_all += loss.item()
        seg_loss_all += seg_loss.item()
        cls_loss_all += cls_loss.item()

    test_loss = loss_all / len(test_dl)
    test_seg_loss = seg_loss_all / len(test_dl)
    test_cls_loss = cls_loss_all / len(test_dl)

    return test_loss, test_seg_loss, test_cls_loss


# train
def train():
    losses = []
    for epoch in range(Epoch):
        loss_all = 0
        seg_loss_all = 0
        cls_loss_all = 0
        model.train()
        for step, (img, mask, label) in enumerate(train_dl):
            if torch.cuda.is_available():
                img, mask, label = img.cuda(), mask.cuda(), label.cuda()

            pred_mask, pred_label = model(img)

            seg_loss = seg_loss_func(pred_mask, mask)
            cls_loss = cls_loss_func(pred_label, label)
            loss = seg_loss + (alpha * cls_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_decay.step()

            loss_all += loss.item()
            seg_loss_all += seg_loss.item()
            cls_loss_all += cls_loss.item()
            logging.info(
                'time:%s | step:%d/%d/%d | loss=%.6f, seg_loss=%.6f, cls_loss=%.6f'
                % (datetime.datetime.now(), step, epoch, Epoch, loss.item(), seg_loss.item(), cls_loss.item()))

        train_loss = loss_all / len(train_dl)
        train_seg_loss = seg_loss_all / len(train_dl)
        train_cls_loss = cls_loss_all / len(train_dl)

        # test
        if epoch % 10 == 0 and epoch != 0:
            test_loss, test_seg_loss, test_cls_loss = test(model)
            print('epoch:%d, test_loss:%f, seg_loss=%.6f, cls_loss=%.6f' % (epoch, test_loss, test_seg_loss, test_cls_loss))
            losses.append([train_loss, train_seg_loss, train_cls_loss, test_loss, test_seg_loss, test_cls_loss])
        else:
            losses.append([train_loss, train_seg_loss, train_cls_loss, '-', '-', '-'])

        # save
        if epoch % 5 == 0 and epoch != 0:
            torch.save(model.state_dict(), './saved_models/multi/net{}-{:.3f}.pth'.format(epoch, train_loss))

    return losses


def save_loss(losses):
    # save train and eval loss
    df = pd.DataFrame(data=losses)
    df.columns = ['loss', 'test']
    df.to_csv('./out/loss/multi:%s.csv' % datetime.datetime.now())


if __name__ == '__main__':
    train_dl = get_dl()
    model, optimizer, lr_decay = get_model()
    seg_loss_func, cls_loss_func = get_loss_fuction()

    losses = train()

    save_loss(losses)
