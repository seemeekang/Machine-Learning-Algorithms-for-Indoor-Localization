import argparse
import logging

import torch
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable

import dataset
import ndf
import pandas as pd
import os
import numpy as np
import pickle as pck
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def parse_arg():
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(asctime)s]: %(levelname)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-dataset', choices=['cold'], default='cold')
    parser.add_argument('-batch_size', type=int, default=32)

    parser.add_argument('-feat_dropout', type=float, default=0.25)

    parser.add_argument('-n_tree', type=int, default=5)
    parser.add_argument('-tree_depth', type=int, default=3)
    parser.add_argument('-n_class', type=int, default=9)
    parser.add_argument('-tree_feature_rate', type=float, default=0.5)

    parser.add_argument('-lr', type=float, default=0.001, help="sgd: 10, adam: 0.001")
    parser.add_argument('-gpuid', type=int, default=0)
    parser.add_argument('-jointly_training', action='store_true', default=True)
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-report_every', type=int, default=10)

    opt = parser.parse_args()
    return opt


def prepare_db(opt):
    print("Use %s dataset" % (opt.dataset))


    if opt.dataset == 'cold':
        dir_name = "/train_std.txt"
        train_df = pd.read_csv(os.getcwd() + dir_name, header=None)
        train_df = train_df.iloc[1:, :]
        train_dataset = dataset.DataGenerator(train_df)
        dir_name = "/val_std.txt"
        eval_df = pd.read_csv(os.getcwd() + dir_name, header=None)
        eval_df = eval_df.iloc[1:, :]
        print(len(eval_df))
        eval_dataset = dataset.DataGenerator(eval_df)
        dir_name = "/test_std.txt"
        test_df = pd.read_csv(os.getcwd() + dir_name, header=None)
        test_df = test_df.iloc[1:, :]
        print(len(test_df))
        test_dataset = dataset.DataGenerator(test_df)
        return {'train': train_dataset, 'eval': eval_dataset,'test':test_dataset}
    else:
        raise NotImplementedError


def prepare_model(opt):
    if opt.dataset == 'cold':
        feat_layer = ndf.ColdFeatureLayer(opt.feat_dropout)
    else:
        raise NotImplementedError

    forest = ndf.Forest(n_tree=opt.n_tree, tree_depth=opt.tree_depth, n_in_feature=feat_layer.get_out_feature_size(),
                        tree_feature_rate=opt.tree_feature_rate, n_class=opt.n_class,
                        jointly_training=opt.jointly_training)
    model = ndf.NeuralDecisionForest(feat_layer, forest)

    if opt.cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    return model


def prepare_optim(model, opt):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=opt.lr, weight_decay=1e-5)


def train(model, optim, db, opt):
    #classs weights for our imbalanced dataset
    ws = [0.837962962962963, 0.5709779179810726, 1.0, 0.11837802485284499, 0.7327935222672064, 0.7903930131004366,
          0.5041782729805013, 0.6830188679245283, 0.5552147239263803]
    ws = torch.FloatTensor(ws).to(device = device)
    for epoch in range(1, opt.epochs + 1):
        # Update \Pi
        if not opt.jointly_training:
            print("Epcho %d : Two Stage Learing - Update PI" % (epoch))
            # prepare feats
            cls_onehot = torch.eye(opt.n_class)
            feat_batches = []
            target_batches = []

            train_loader = torch.utils.data.DataLoader(db['train'], batch_size=opt.batch_size, shuffle=False)
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(train_loader):
                    if opt.cuda:
                        data, target, cls_onehot = data.cuda(), target.cuda(), cls_onehot.cuda()
                    data = Variable(data)
                    # Get feats
                    feats = model.feature_layer(data)
                    feats = feats.view(feats.size()[0], -1)
                    feat_batches.append(feats)
                    target_batches.append(cls_onehot[target.type(torch.LongTensor)])

                # Update \Pi for each tree
                for tree in model.forest.trees:
                    mu_batches = []
                    for feats in feat_batches:
                        mu = tree(feats)  # [batch_size,n_leaf]
                        mu_batches.append(mu)
                    for _ in range(20):
                        new_pi = torch.zeros((tree.n_leaf, tree.n_class))  # Tensor [n_leaf,n_class]
                        if opt.cuda:
                            new_pi = new_pi.cuda()
                        for mu, target in zip(mu_batches, target_batches):
                            pi = tree.get_pi()  # [n_leaf,n_class]
                            prob = tree.cal_prob(mu, pi)  # [batch_size,n_class]

                            # Variable to Tensor
                            pi = pi.data
                            prob = prob.data
                            mu = mu.data

                            _target = target.unsqueeze(1)  # [batch_size,1,n_class]
                            _pi = pi.unsqueeze(0)  # [1,n_leaf,n_class]
                            _mu = mu.unsqueeze(2)  # [batch_size,n_leaf,1]
                            _prob = torch.clamp(prob.unsqueeze(1), min=1e-6, max=1.)  # [batch_size,1,n_class]

                            _new_pi = torch.mul(torch.mul(_target, _pi), _mu) / _prob  # [batch_size,n_leaf,n_class]
                            new_pi += torch.sum(_new_pi, dim=0)
                        # test
                        # import numpy as np
                        # if np.any(np.isnan(new_pi.cpu().numpy())):
                        #    print(new_pi)
                        # test
                        new_pi = F.softmax(Variable(new_pi), dim=1).data
                        tree.update_pi(new_pi)

        # Update \Theta
        model.train()

        train_loader = torch.utils.data.DataLoader(db['train'], batch_size=opt.batch_size, shuffle=True)
        for batch_idx, (data, target) in enumerate(train_loader):
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optim.zero_grad()
            output = model(data)



            target_temp = target.cpu().numpy()
            target_indices = np.array([np.where(r == 1)[0][0] for r in target_temp])
            target_indices = torch.from_numpy(target_indices).type(torch.LongTensor)

            loss = F.nll_loss(torch.log(output), target_indices.to(device,dtype = torch.long),weight=ws)
            loss.backward()
            # torch.nn.utils.clip_grad_norm([ p for p in model.parameters() if p.requires_grad],
            #                              max_norm=5)
            optim.step()
            if batch_idx % opt.report_every == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

        # Eval
        model.eval()
        test_loss = 0
        correct = 0
        test_loader = torch.utils.data.DataLoader(db['eval'], batch_size=opt.batch_size, shuffle=False)
        cnt = 0
        pred_arr = np.zeros(shape=(922,9))
        with torch.no_grad():
            for data, target in test_loader:
                if opt.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                output = model(data)
                target_temp = target.cpu().numpy()
                target_indices = np.array([np.where(r == 1)[0][0] for r in target_temp])
                target_indices = torch.from_numpy(target_indices).type(torch.LongTensor)
                test_loss += F.nll_loss(torch.log(output), target_indices.type(torch.LongTensor).to(device = device), size_average=False,weight=ws).item()  # sum up batch loss


                if cnt == int(np.floor(922/opt.batch_size)):
                    pred_arr[cnt * opt.batch_size:, :] = output.cpu().numpy()
                else:
                    pred_arr[cnt*opt.batch_size :(cnt+1)*opt.batch_size,:] = output.cpu().numpy()

                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target_indices.data.type(torch.LongTensor).to(device = device).view_as(pred)).cpu().sum()
                cnt = cnt+1

            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f})\n'.format(
                test_loss, correct, len(test_loader.dataset),
                correct / len(test_loader.dataset)))
        pck.dump(pred_arr, open(os.getcwd() + "/preds/predeval{}correct{}.npy".format(epoch,correct), "wb"))

        test_loss = 0
        correct = 0
        test_loader = torch.utils.data.DataLoader(db['test'], batch_size=opt.batch_size, shuffle=False)
        cnt = 0
        pred_arr = np.zeros(shape=(3672, 9))
        with torch.no_grad():
            for data, target in test_loader:
                if opt.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                output = model(data)
                target_temp = target.cpu().numpy()
                target_indices = np.array([np.where(r == 1)[0][0] for r in target_temp])
                target_indices = torch.from_numpy(target_indices).type(torch.LongTensor)
                test_loss += F.nll_loss(torch.log(output), target_indices.type(torch.LongTensor).to(device=device),
                                        size_average=False,weight=ws).item()  # sum up batch loss

                if cnt == int(np.floor(3672 / opt.batch_size)):
                    pred_arr[cnt * opt.batch_size:, :] = output.cpu().numpy()
                else:
                    pred_arr[cnt * opt.batch_size:(cnt + 1) * opt.batch_size, :] = output.cpu().numpy()

                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(
                    target_indices.data.type(torch.LongTensor).to(device=device).view_as(pred)).cpu().sum()
                cnt = cnt + 1

            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f})\n'.format(
                test_loss, correct, len(test_loader.dataset),
                correct / len(test_loader.dataset)))
        pck.dump(pred_arr, open(os.getcwd() + "/preds/predtest{}correct{}.npy".format(epoch, correct), "wb"))


def main():
    opt = parse_arg()

    # GPU
    opt.cuda = opt.gpuid >= 0
    if opt.gpuid >= 0:
        torch.cuda.set_device(opt.gpuid)
    else:
        print("WARNING: RUN WITHOUT GPU")

    db = prepare_db(opt)
    model = prepare_model(opt)
    optim = prepare_optim(model, opt)
    train(model, optim, db, opt)


if __name__ == '__main__':
    main()
