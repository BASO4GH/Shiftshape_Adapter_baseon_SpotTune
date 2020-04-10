#main_updated.py
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import time
import argparse
import numpy as np
import json
import collections

import imdbfolder as imdbfolder
from spottune_models import *
import models
import agent_net

from utils import *
from gumbel_softmax import *

parser = argparse.ArgumentParser(description='PyTorch SpotTune')

#parser.add_argument('--nb_epochs', default=110, type=int, help='nb epochs') #original
parser.add_argument('--nb_epochs', default=31, type=int, help='nb epochs')

parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate of net')
parser.add_argument('--lr_agent', default=0.01, type=float, help='initial learning rate of agent')

parser.add_argument('--datadir', default='./data/decathlon-1.0/', help='folder containing data folder')
parser.add_argument('--imdbdir', default='./data/decathlon-1.0/annotations/', help='annotation folder')
parser.add_argument('--ckpdir', default='./cv/', help='folder saving checkpoint')

parser.add_argument('--seed', default=0, type=int, help='seed')

parser.add_argument('--step1', default=40, type=int, help='nb epochs before first lr decrease')
parser.add_argument('--step2', default=60, type=int, help='nb epochs before second lr decrease')
parser.add_argument('--step3', default=80, type=int, help='nb epochs before third lr decrease')

args = parser.parse_args()
''' original
weight_decays = [
    ("aircraft", 0.0005),
    ("cifar100", 0.0),
    ("daimlerpedcls", 0.0005),
    ("dtd", 0.0),
    ("gtsrb", 0.0),
    ("omniglot", 0.0005),
    ("svhn", 0.0),
    ("ucf101", 0.0005),
    ("vgg-flowers", 0.0001),
    ("imagenet12", 0.0001)]

datasets = [
    ("aircraft", 0),
    ("cifar100", 1),
    ("daimlerpedcls", 2),
    ("dtd", 3),
    ("gtsrb", 4),
    ("omniglot", 5),
    ("svhn", 6),
    ("ucf101", 7),
    ("vgg-flowers", 8)]
'''

weight_decays = [
    ("aircraft", 0.0005),
    ("dtd", 0.0),
    ("ucf101", 0.0005),
    ("vgg-flowers", 0.0001)]
datasets = [
    ("aircraft", 0),
    ("dtd", 1),
    ("ucf101", 2),
    ("vgg-flowers", 3)]


# weight_decays = [
#     ("vgg-flowers", 0.0001),
#     ("imagenet12", 0.0001)]
# datasets = [
#     ("vgg-flowers", 0)]

# weight_decays = [
#     ("omniglot", 0.0005)]
# datasets = [
#     ("omniglot", 0)]


datasets = collections.OrderedDict(datasets)
weight_decays = collections.OrderedDict(weight_decays)

with open(args.ckpdir + '/weight_decays.json', 'w') as fp:
    json.dump(weight_decays, fp)
#Changed pock to epock here
def train(dataset, epoch, train_loader, net, agent, net_optimizer, agent_optimizer):
    #Train the model
    global logFileName
    fLog = open(logFileName,'a')
    net.train()
    agent.train()

    total_step = len(train_loader)
    tasks_top1 = AverageMeter()
    tasks_losses = AverageMeter()

    for i, task_batch in enumerate(train_loader):
        images = task_batch[0] 
        labels = task_batch[1]    

        if use_cuda:
            images, labels = images.cuda(async=True), labels.cuda(async=True)
        images, labels = Variable(images), Variable(labels)	   

        probs = agent(images)

        action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
        policy = action[:,:,1]

        outputs = net.forward(images, policy)
        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(labels.data).cpu().sum()
        tasks_top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))

        # Loss
        loss = criterion(outputs, labels)
        tasks_losses.update(loss.item(), labels.size(0))
      
        if i % 50 == 0:
            print(("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc Val: {:.4f}, Acc Avg: {:.4f}"
                .format(epoch+1, args.nb_epochs, i+1, total_step, tasks_losses.val, tasks_top1.val, tasks_top1.avg)))
            
            fLog.write("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc Val: {:.4f}, Acc Avg: {:.4f}"
                .format(epoch+1, args.nb_epochs, i+1, total_step, tasks_losses.val, tasks_top1.val, tasks_top1.avg)+'\n')
       
        #---------------------------------------------------------------------#
        # Backward and optimize
        net_optimizer.zero_grad()
        agent_optimizer.zero_grad()

        loss.backward()  
        net_optimizer.step()    #original
        agent_optimizer.step()    #original
        # net_optimizer.step(loss)    #adapter
        # agent_optimizer.step(loss)    #adapter
                   
    fLog.close()        
    return tasks_top1.avg , tasks_losses.avg

def test(epoch, val_loader, net, agent, dataset):
    global logFileName
    net.eval()
    agent.eval()

    tasks_top1 = AverageMeter()
    tasks_losses = AverageMeter() 

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if use_cuda:
                images, labels = images.cuda(async=True), labels.cuda(async=True)
            images, labels = Variable(images), Variable(labels)

       	    probs = agent(images)
            action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
            policy = action[:,:,1]
            outputs = net.forward(images, policy)

            _, predicted = torch.max(outputs.data, 1)
            correct = predicted.eq(labels.data).cpu().sum()
            tasks_top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))
        
            # Loss
            loss = criterion(outputs, labels)
            tasks_losses.update(loss.item(), labels.size(0))           
    fLog = open(logFileName,'a')
    fLog.write("test accuracy:")
    fLog.write("Epoch [{}/{}], Loss: {:.4f}, Acc Val: {:.4f}, Acc Avg: {:.4f}"
               .format(epoch+1, args.nb_epochs, tasks_losses.avg, tasks_top1.val, tasks_top1.avg)+'\n')
    fLog.close()
    print("test accuracy:")
    print(("Epoch [{}/{}], Loss: {:.4f}, Acc Val: {:.4f}, Acc Avg: {:.4f}"
        .format(epoch+1, args.nb_epochs, tasks_losses.avg, tasks_top1.val, tasks_top1.avg)))

    return tasks_top1.avg, tasks_losses.avg

def load_weights_to_flatresnet(source, net, num_class, dataset):
    checkpoint = torch.load(source, encoding='latin1')
    #checkpoint={ key.decode(): val for key, val in checkpoint.items() }
    net_old = checkpoint['net']

    store_data = []
    t = 0
    for name, m in net_old.named_modules():
        # print('old ms:'+str(type(m)))
        if isinstance(m, nn.Conv2d):
            store_data.append(m.weight.data)
            t += 1

    element = 0
    for name, m in net.named_modules():
        # print('new ms:'+str(type(m)))
        if isinstance(m, nn.Conv2d) and 'parallel_blocks' not in name:
            m.weight.data = torch.nn.Parameter(store_data[element].clone())
            element += 1

    element = 1
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and 'parallel_blocks' in name:
            m.weight.data = torch.nn.Parameter(store_data[element].clone())
            element += 1

    store_data = []
    store_data_bias = []
    store_data_rm = []
    store_data_rv = []
    for name, m in net_old.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            store_data.append(m.weight.data)
            store_data_bias.append(m.bias.data)
            store_data_rm.append(m.running_mean)
            store_data_rv.append(m.running_var)

    element = 0
    for name, m in net.named_modules():
        # print('name: '+name+'||type: '+str(type(m)))
        if isinstance(m, nn.BatchNorm2d) and 'parallel_block' not in name:
            m.weight.data = torch.nn.Parameter(store_data[element].clone())
            m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
            m.running_var = store_data_rv[element].clone()
            m.running_mean = store_data_rm[element].clone()
            element += 1

    element = 1
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d) and 'parallel_block' in name:
            m.weight.data = torch.nn.Parameter(store_data[element].clone())
            m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
            m.running_var = store_data_rv[element].clone()
            m.running_mean = store_data_rm[element].clone()
            element += 1
        if 'adapter' in name: #ongoingversion
          m.adjustTensor = torch.nn.Parameter(torch.rand(10,10).cuda())
    
    del net_old
    return net

def get_model(model, num_class, dataset = None):
  if model == 'resnet26':
      rnet = resnet26(num_class)
      if dataset is not None:
          if dataset == 'imagenet12':
            source = './resnet26_pretrained.t7'
      else:
            source = './cv/' + dataset + '/' + dataset + '.t7'
      rnet = load_weights_to_flatresnet(source, rnet, num_class, dataset)
  return rnet

def paramSizefromList(n):
  res = 1
  for i in n:
    res*=i
  return res

#####################################
# Prepare data loaders
train_loaders, val_loaders, num_classes = imdbfolder.prepare_data_loaders(list(datasets.keys()), args.datadir, args.imdbdir, True)
criterion = nn.CrossEntropyLoss()

logFileName = 'log/log_'+ str(time.strftime('%y-%m-%d_%H:%M:%S_%Z',time.localtime(time.time())))+'.txt'
fLog = open(logFileName,'w') #record results
fLog.close()

oaTraAcc, oaTraLoss, oaTesAcc, oaTesLoss = [],[],[],[]
overallParamList = []
curParamList = []
allParamList = []

for i, dataset in enumerate(datasets.keys()):
    fLog = open(logFileName,'a')
    fLog.write(dataset+'---------------------------------------------------------'+'\n')
    fLog.close()
    print(dataset +'--------------------')
    pretrained_model_dir = args.ckpdir + dataset

    if not os.path.isdir(pretrained_model_dir):
        os.mkdir(pretrained_model_dir)

    results = np.zeros((4, args.nb_epochs, len(num_classes)))
    f = pretrained_model_dir + "/params.json"
    with open(f, 'w') as fh:
        json.dump(vars(args), fh)     
  
    num_class = num_classes[datasets[dataset]]
    
    net = get_model("resnet26", num_class, dataset = "imagenet12")
	
	
    agent = agent_net.resnet(sum(net.layer_config) * 2)
	
    # freeze the original blocks
    flag = True
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and 'parallel_blocks' not in name:
            if flag is True:
                flag = False
            else:
                m.weight.requires_grad = False
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
        agent.cuda()

        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
        #net = nn.DataParallel(net)
        #agent = nn.DataParallel(agent)

    '''see params in resnet'''
    # unnamed,named= 0,0
    # for param in net.parameters():
    #     unnamed+=1
    #     if param.requires_grad:
    #       print (name+': '+str(param.data.size())+'(req_g)++++++++++')
    #     else:
    #       print (name+': '+str(param.data.size()))
    # for name, param in net.named_parameters():
    #     named+=1
    #     if param.requires_grad:
    #       print (name+': '+str(param.data.size())+'(req_g)')
    #     else:
    #       print (name+': '+str(param.data.size()))
    # print('unnamed: '+str(unnamed))
    # print('named: '+str(named))
    '''see params in resnet'''

    allRGParam = []
    allParam = []
    #calculate orinally all reqGrad param and bypass bn1 conv2
    for name, param in net.named_parameters():
        if 'adapter' not in name:
          allParam.append(paramSizefromList(list(param.data.size())))
        if param.requires_grad:
            allRGParam.append(paramSizefromList(list(param.data.size())))
        if 'parallel' in name and 'conv2' in name or 'bn1' in name: #ongoingversion
          param.requires_grad = False  #ongoingversion

    # oriAllReqGradParam = 5847556 #without adapters
    print(('all parameters in this dataset: '+str(np.sum(allParam))))
    curRGParam = []
    for name, param in net.named_parameters():
        if param.requires_grad:
          print((name+': '+str(param.data.size())+'(req_g)'))
          curRGParam.append(paramSizefromList(list(param.data.size())))
        else:
          print((name+': '+str(param.data.size())))

    overallParamList.append(np.sum(allRGParam))
    curParamList.append(np.sum(curRGParam))
    allParamList.append(np.sum(allParam))
    # print(allReqGradParam)
    # allReqNum = sum(allReqGradParam)
    # print(allReqNum)
    # perc = float(allReqNum/5847556)
    # print(perc)
    # print('current param percentage: '+str(perc))
    ''''''

    optimizer = optim.SGD([p for p in net.parameters() if p.requires_grad], lr= args.lr, momentum=0.9, weight_decay= weight_decays[dataset])
    agent_optimizer = optim.SGD(agent.parameters(), lr= args.lr_agent, momentum= 0.9, weight_decay= 0.001)

    start_epoch = 0
    avgTimeDur = []
    fLog = open(logFileName,'a')
    trainAccList, trainLossList = [],[]
    testAccList, testLossList = [],[]
    for epoch in range(start_epoch, start_epoch+args.nb_epochs):
        adjust_learning_rate_net(optimizer, epoch, args)
        adjust_learning_rate_agent(agent_optimizer, epoch, args)

        st_time = time.time()
        train_acc, train_loss = train(dataset, epoch, train_loaders[datasets[dataset]], net, agent, optimizer, agent_optimizer)
        test_acc, test_loss = test(epoch, val_loaders[datasets[dataset]], net, agent, dataset)

        if epoch>=args.nb_epochs*0.9:
          trainAccList.append(train_acc)
          trainLossList.append(train_loss)
          testAccList.append(test_acc)
          testLossList.append(test_loss)

        # Record statistics
        results[0:2,epoch,i] = [train_loss, train_acc]
        results[2:4,epoch,i] = [test_loss,test_acc]

        curTime = time.time()
        print(('Epoch lasted {0}'.format(curTime-st_time)))
        
        avgTimeDur.append(curTime-st_time)
        fLog.write('Epoch lasted {0}'.format(curTime-st_time)+'\n')
    avgTvalue = (sum(avgTimeDur)-max(avgTimeDur)-min(avgTimeDur))/(len(avgTimeDur)-2)
    print(('Avg Time lasted for each Epoch {0}'.format(avgTvalue)+'\n'))
    fLog.write('Avg Time lasted for each Epoch {0}'.format(avgTvalue)+'\n')
    fLog.write('EndTrainAvrAccuracy: {:.4f}; EndTestAvrAccuracy: {:.4f}; EndTrainAvrLoss: {:.4f}; EndTestAvrLoss: {:.4f}'
        .format(np.mean(trainAccList), np.mean(testAccList), np.mean(trainLossList), np.mean(testLossList))+'\n')
    fLog.close()
    oaTraAcc.append(np.mean(trainAccList))
    oaTraLoss.append(np.mean(np.mean(trainLossList)))
    oaTesAcc.append(np.mean(testAccList))
    oaTesLoss.append(np.mean(testLossList))

    state = {
        'net': net,
        'agent': agent,
    }

    torch.save(state, pretrained_model_dir +'/withAdapter_' + dataset + '.t7')
    np.save(pretrained_model_dir + '/withAdapter_statistics', results)
    print((pretrained_model_dir +'/withAdapter_' + dataset + '.t7'+' Generated'))
print(('# of all param: '+str(np.sum(allParamList))))
print(('# of all require grad param: '+str(np.sum(overallParamList))))
print(('# of current param: '+str(np.sum(curParamList))))
print(('percentage: '+str(np.sum(curParamList)/np.sum(overallParamList))))

fLog = open(logFileName,'a')
fLog.write('overall statics: '+'\n')
fLog.write('overallTrainAvrAccuracy: {:.4f}; overallTestAvrAccuracy: {:.4f}; overallTrainAvrLoss: {:.4f}; overallTestAvrLoss: {:.4f}'
        .format(np.mean(oaTraAcc), np.mean(oaTesAcc), np.mean(oaTraLoss), np.mean(oaTesLoss))+'\n')
fLog.close()
fSta = open('log/drawStatistic.txt','a')
fSta.write('overall statics:(order: trainAcc, testAcc, trainLoss, testLoss)'+'\n')
fSta.write(str(oaTraAcc)+'\n')
fSta.write(str(oaTesAcc)+'\n')
fSta.write(str(oaTraLoss)+'\n')
fSta.write(str(oaTesLoss)+'\n')
fSta.write('parameters: '+'\n')
fSta.close()
