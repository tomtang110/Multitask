# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from util import to_categorical
def cal_acc(pred,true,is_torch=True):
    if is_torch:
        pred = torch.max(pred.data, 1)[1].cpu()
        true = true.data.cpu()
        acc = metrics.accuracy_score(true, pred)
    else:
        pred = np.argmax(pred,axis=1)
        acc = metrics.accuracy_score(true, pred)
    return acc

def cal_auc(pred,true,is_torch=True,is_softmax=False,isbin = True):
    if is_torch:
        true = true.data.cpu().numpy()
        pred = pred.data.cpu().numpy()
    if is_softmax:
        pred = np.argmax(pred, axis=1)
    if isbin:
        true = to_categorical(true,2)
    # trues = to_categorical(true,2)
    auc1 = metrics.roc_auc_score(true,pred)
    auc2 = metrics.roc_auc_score(1-true,1-pred)
    auc = (auc1+auc2)/2
    return auc

def train(config, model, train_iter=None, dev_iter=None, test_iter=None):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    total_batch = 0  # 记录进行到多少batch
    dev_best = float('-inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            outputs,regul = model(trains)
            outputs1,outputs2 = outputs[0].view(-1),outputs[1].view(-1)
            label1,label2 = labels[0],labels[1]
            model.zero_grad()
            loss1 = config.loss_fn(outputs1,label1,reduction='mean')
            loss2 = config.loss_fn(outputs2,label2 ,reduction='mean')

            loss = loss1+ loss2
            loss.backward()
            optimizer.step()

            if total_batch % 10 == 0:
                # 每多少轮输出在训练集和验证集上的效果

                train_auc1 = cal_auc(outputs1,label1,isbin=False)

                train_auc2 = cal_auc(outputs2,label2,isbin=False)

                # dev_acc1,dev_acc2,dev_auc1,dev_auc2, dev_loss1,dev_loss2 = evaluate(model, dev_iter)

                dev_auc1, dev_auc2, dev_loss1, dev_loss2 = evaluate(config,model, dev_iter)
                # dev_acc = (dev_acc1+dev_acc2)/2
                dev_auc = (dev_auc1+dev_auc2)/2
                if dev_auc > dev_best:
                    dev_best = dev_auc
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                # msg = 'Iter: {0:>6},  Train Loss1: {1:>5.2},  Train Loss2: {2:>5.2}, ' \
                #       'Train Acc1: {3:>6.2%},  Train Acc2: {4:>6.2%},  ' \
                #       'Val Loss1: {5:>5.2},  Val Loss2: {6:>5.2},' \
                #       'Val Acc1: {7:>6.2%}, Val Acc2: {8:>6.2%}, Val Auc1: {9:>6.2%}, Val Auc2: {10:>6.2%},  {11}'
                # print(msg.format(total_batch, loss1.item(), loss2.item(),
                #                  train_acc1,train_acc2, dev_loss1,dev_loss2,
                #                  dev_acc1,dev_acc2,dev_auc1,dev_auc2, improve))

                msg = 'Iter: {0:>6},  Train Loss1: {1:>5.2},  Train Loss2: {2:>5.2}, ' \
                      'Train Auc1: {3:>6.2%},  Train Auc2: {4:>6.2%},  ' \
                      'Val Loss1: {5:>5.2},  Val Loss2: {6:>5.2},' \
                      'Val Auc1: {7:>6.2%}, Val Auc2: {8:>6.2%},  {9}'
                print(msg.format(total_batch, loss1.item(), loss2.item(),
                                 train_auc1,train_auc2, dev_loss1,dev_loss2,
                                 dev_auc1,dev_auc2, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break

        if flag:
            break

    test(config,model, test_iter)



def test(config,model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()

    # acc1, acc2,auc1,auc2,loss1,loss2 = evaluate(model, test_iter, test=True)

    auc1, auc2, loss1, loss2 = evaluate(config,model, test_iter, test=True)
    # msg = 'Test Loss1: {0:>5.2},  Test  Loss1: {1:>5.2},' \
    #       'Test  Acc: {2:>6.2%}, Test  Acc: {3:>6.2%},Test  Auc: {4:>6.2%}, Test  Auc: {5:>6.2%}'
    # print(msg.format(loss1.item(), loss2.item(),
    #                  acc1,acc2,auc1,auc2))

    msg = 'Test Loss1: {0:>5.2},  Test  Loss1: {1:>5.2},' \
              'Test  Auc: {2:>6.2%}, Test  Auc: {3:>6.2%}'
    print(msg.format(loss1.item(), loss2.item(),
                     auc1,auc2))
def evaluate(config,model, data_iter, test=False):
    model.eval()
    predict_all1 = []
    labels_all1 = []
    predict_all2 = []
    labels_all2 = []
    loss_t1 = 0
    loss_t2 = 0
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs,regul= model(texts)
            outputs1, outputs2 = outputs[0], outputs[1]
            label1, label2 = labels[0], labels[1]
            loss1 = config.loss_fn(outputs1.view(-1), label1,reduction='mean')
            loss2 = config.loss_fn(outputs2.view(-1),label2,reduction='mean')
            loss_t1 += loss1
            loss_t2 += loss2
            if predict_all1 == []:
                predict_all1 = outputs1.cpu().numpy()
                predict_all2 = outputs2.cpu().numpy()
                labels_all1 = label1.cpu().numpy()
                labels_all2 = label2.cpu().numpy()
            else:
                predict_all1 = np.concatenate((predict_all1, outputs1.cpu().numpy()), axis=0)
                predict_all2 = np.concatenate((predict_all2, outputs2.cpu().numpy()), axis=0)
                labels_all1 = np.concatenate((labels_all1, label1.cpu().numpy()), axis=0)
                labels_all2 = np.concatenate((labels_all2, label2.cpu().numpy()), axis=0)


    # acc1 = cal_acc(predict_all1,labels_all1,False)
    # acc2 = cal_acc(predict_all2,labels_all2,False)
    auc1 = cal_auc(predict_all1,labels_all1,False,isbin=False)
    auc2 = cal_auc(predict_all2,labels_all2,False,isbin=False)

    return auc1,auc2,loss_t1/len(data_iter),loss_t2/len(data_iter)
