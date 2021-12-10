import time, datetime
import pickle
import argparse
import yaml

from pathlib import Path
import torch
import torch.optim as optim
import numpy as np
from torchvision import transforms
from loader import AV_Dataset
from model import SimpleImageCorrNet

parser = argparse.ArgumentParser(description="train unsupervised model")
parser.add_argument("conf", type=str)
parser.add_argument("dataset_path", type=str)
parser.add_argument("workdir", type=str)
opts = parser.parse_args()


def update_args(yamlpath, opts):
    opts = vars(opts) #Namespace -> dict
    with open(yamlpath, "r") as f:
        conf = yaml.safe_load(f)
    assert set(opts.keys()).isdisjoint(set(conf.keys())) #opts & conf == {}
    opts.update(conf) #opts += conf
    return argparse.Namespace(**opts) #dict -> Namespace

opts = update_args(opts.conf, opts)

LOG_NAME=opts.workdir + "/log/" + datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H_%M_%S") + ".log"
Path(LOG_NAME).parent.mkdir(parents=True)

def logger(s):
    print(s, flush=True)
    with open(LOG_NAME, "a") as f:
        f.write(s)
        f.write("\n")


class KeepUnderThr(object):
    def __init__(self, times, thr):
        self.times = times
        self.thr = thr
        self.count = 0

    def __call__(self, loss):
        if loss < self.thr:
            self.count += 1
        else:
            self.count = 0
        return self.count >= self.times


def get_remainder(datasize, b_batch_size, batch_size):
    assert b_batch_size % batch_size == 0
    remainder = datasize % b_batch_size
    rem_start_iter = (datasize // b_batch_size) * (b_batch_size // batch_size)
    return remainder, rem_start_iter


def train_video_corr_net(model, device, loader, optimizer, epoch, start_time, break_fn):
    from loss import sim_loss_batchsum
    model.train()
    sum_loss = 0
    batch_num = 0
    iter_num = 0
    NUM_B_BATCH = opts.batch_size*10
    rem, rem_start_iter = get_remainder(len(loader.dataset), NUM_B_BATCH, loader.batch_size)
    for batch_idx, (snd, img) in enumerate(loader):
        img, snd = img.to(device), snd.to(device)
        loss = 0
        if batch_num == 0:
            iter_mean_loss = 0
            optimizer.zero_grad()
        vcs, acs = model(img, snd)
        loss = sim_loss_batchsum(vcs,acs,rho=opts.rho)
        if batch_idx < rem_start_iter:
            loss /= NUM_B_BATCH
        else:
            loss /= rem
        iter_mean_loss += loss.item()
        batch_num += img.shape[0]
        loss.backward()
        if batch_idx+1 != len(loader) and batch_num != NUM_B_BATCH: continue
        iter_num += 1
        sum_loss += iter_mean_loss
        optimizer.step()

        logger(f"Train Epoch: {epoch} [{(batch_idx+1)*len(img)}/{len(loader.dataset)} ({int(100.*(batch_idx+1)/len(loader)):3}%)]\tLoss: {iter_mean_loss:.6f}\tTime:{time.time() - start_time}")
        batch_num = 0
    loss_mean = sum_loss/iter_num
    logger("Train Epoch: {} Loss: {:.6f}\tTime:{}".format(
            epoch, loss_mean,time.time() - start_time))
    return loss_mean


def repeat_last(batch):
    import numpy as np
    a, v = [], []
    a_len = -1
    for b in batch:
        a.append(b[0])
        v.append(b[1])
        a_len = max(a_len, len(b[0]))
    a_pad = []
    for ae in a:
        pad = ae[-1:].repeat(a_len-len(ae), axis=0)
        ae = np.concatenate((ae, pad), axis=0)
        a_pad.append(torch.from_numpy(ae))
    a_pad = torch.stack(a_pad, dim=0).float()
    v = torch.stack(v, dim=0).float()
    return a_pad,v


def train_data2(shuffle=True, dataset_path="data/plist_train_noise.txt"):
    from torch.utils.data import DataLoader
    paths = []
    with open(dataset_path, "r") as f:
        for t in f:
            paths.append(tuple(t.strip().split()))
    v_tr = transforms.ToTensor()
    v_tr_train = transforms.ToTensor()
    loader = DataLoader(AV_Dataset(paths, None, v_tr_train), batch_size=opts.batch_size, shuffle=shuffle, num_workers=7, pin_memory=True, collate_fn=repeat_last)
    return loader


def saves(model, optimizer, path):
    stt = {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict()
            }
    torch.save(stt, path)


if __name__ == "__main__":
    LOAD_EPOCH=0
    END_EPOCH=opts.epoch
    loader = train_data2(dataset_path=opts.dataset_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ext_fusion = True if opts.ext_fusion == 1 else False
    del_last = True if opts.del_last == 1 else False
    model = SimpleImageCorrNet(num_class = 50).to(device)
    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start_time = time.time()
    break_fn = KeepUnderThr(times=10, thr=1e-5)
    loss_list = []
    patience = 0
    loss_min = 100000
    for epoch in range(LOAD_EPOCH+1, END_EPOCH+1):
        loss_h = train_video_corr_net(model, device, loader, optimizer, epoch, start_time, break_fn)
        len_loader = len(loader)
        loss_list.append(loss_h)
        np.save(opts.workdir + "/unsup_loss.npy", loss_list)
        loss_min = min(loss_list)
        
        # if epoch%25 == 0:
        #     saves(model, optimizer, f"{opts.workdir}/unsup_backend_{epoch}epoch.pth.tar")
        #     print("save")

        if loss_h > loss_min:
            patience += 1
        else:
            patience = 0
            saves(model, optimizer, f"{opts.workdir}/unsup_backend.pth.tar")
            print("save")
        if patience > 20*7200//opts.batch_size//len_loader and lr > 1e-7:
            lr = lr/10
            patience = 0
            optimizer = optim.Adam(model.parameters(), lr=lr)
            print("Decrease lr")
        if patience > 100*7200//opts.batch_size//len_loader:
            print("Early stop", loss_min)
            break
        print(loss_min)