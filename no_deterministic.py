import argparse
import sys
import os
import time
import random
import numpy
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as torchvision_models
import torchnet as tnt

from tqdm import tqdm
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler

import models
from folder import ImageFolder

cudnn.benchmark = False
cudnn.deterministic = True

numpy.set_printoptions(formatter={'float': '{:0.8f}'.format})
torch.set_printoptions(precision=8)

dataset_names = ('mnist', 'cifar10', 'cifar100', 'imagenet2012')

local_model_names = sorted(name for name in models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(models.__dict__[name]))
remote_model_names = sorted(name for name in torchvision_models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(torchvision_models.__dict__[name]))

parser = argparse.ArgumentParser(description='Train')

parser.add_argument('-dir', '--artifact-dir', type=str, metavar='DIR',
                    help='the project directory')
parser.add_argument('-data', '--dataset-dir', type=str, metavar='DATA',
                    help='output dir for logits extracted')
parser.add_argument('-x', '--executions', default=5, type=int, metavar='N',
                    help='Number of executions (default: 5)')
parser.add_argument('-d', '--dataset', metavar='DATA', default=None, choices=dataset_names,
                    help='dataset to be used: ' + ' | '.join(dataset_names))
parser.add_argument('-lm', '--local-model', metavar='MODEL', default=None, choices=local_model_names,
                    help='model to be used: ' + ' | '.join(local_model_names))
parser.add_argument('-rm', '--remote-model', metavar='MODEL', default=None, choices=remote_model_names,
                    help='model to be used: ' + ' | '.join(remote_model_names))
parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-e', '--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-bs', '--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128)')
parser.add_argument('-tss', '--train-set-split', default=0.9, type=float, metavar='VS',
                    help='fraction of trainset to be used to validation')
parser.add_argument('-lr', '--original-learning-rate', default=0.1, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('-lrdr', '--learning-rate-decay-rate', default=0.2, type=float, metavar='LRDR',
                    help='learning rate decay rate')
parser.add_argument('-lrdp', '--learning-rate-decay-period', default=30, type=int, metavar='LRDP',
                    help='learning rate decay period')
parser.add_argument('-lrde', '--learning-rate-decay-epochs', default="60 80 90", metavar='LRDE',
                    help='learning rate decay epochs')
parser.add_argument('-exps', '--experiments', default="baseline", type=str, metavar='EXPERIMENTS',
                    help='Experiments to be performed')
parser.add_argument('-mm', '--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-wd', '--weight-decay', default=5*1e-4, type=float, metavar='W',
                    help='weight decay (default: 5*1e-4)')
parser.add_argument('-pf', '--print-freq', default=16, type=int, metavar='N',
                    help='print frequency (default: 16)')
parser.add_argument('-tr', '--train', const=True, nargs='?', type=bool,
                    help='if true, train the model')
parser.add_argument('-el', '--extract-logits', const=True, nargs='?', type=bool,
                    help='if true, extract logits')

args = parser.parse_args()
args.learning_rate_decay_epochs = sorted([int(item) for item in args.learning_rate_decay_epochs.split()])
args.experiments = args.experiments.split("_")


def execute(execution_path, writer):
    print(cudnn.benchmark)
    print(cudnn.deterministic)

    if args.dataset == "mnist":
        args.number_of_dataset_classes = 10
        dataset_path = args.dataset_dir if args.dataset_dir else "../datasets/mnist/images"
        normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
        train_transform = transforms.Compose(
            [transforms.ToTensor(), normalize])
        inference_transform = transforms.Compose([transforms.ToTensor(), normalize])
    elif args.dataset == "cifar10":
        args.number_of_dataset_classes = 10
        dataset_path = args.dataset_dir if args.dataset_dir else "../datasets/cifar10/images"
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261])
        train_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), normalize])
        inference_transform = transforms.Compose([transforms.ToTensor(), normalize])
    elif args.dataset == "cifar100":
        args.number_of_dataset_classes = 100
        dataset_path = args.dataset_dir if args.dataset_dir else "../datasets/cifar100/images"
        normalize = transforms.Normalize(mean=[0.507, 0.486, 0.440], std=[0.267, 0.256, 0.276])
        train_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), normalize])
        inference_transform = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        args.number_of_dataset_classes = 1000
        dataset_path = args.dataset_dir if args.dataset_dir else "../datasets/imagenet2012/images"
        if args.arch.startswith('inception'):
            size = (299, 299)
        else:
            size = (224, 256)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose(
            [transforms.RandomSizedCrop(size[0]),  # 224 , 299
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), normalize])
        inference_transform = transforms.Compose(
            [transforms.Scale(size[1]),  # 256
             transforms.CenterCrop(size[0]),  # 224 , 299
             transforms.ToTensor(), normalize])

    args.normal_classes = sorted(random.sample(range(0, args.number_of_dataset_classes), args.number_of_model_classes))
    print("NORMAL CLASSES:\t", args.normal_classes)

    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'val')

    # Creating sets...
    train_set = ImageFolder(train_path, transform=train_transform, selected_classes=args.normal_classes,
                            target_transform=args.normal_classes.index)
    val_set = ImageFolder(train_path, transform=inference_transform, selected_classes=args.normal_classes,
                          target_transform=args.normal_classes.index)

    # Preparing train and validation samplers...
    total_examples = {}
    for index in range(len(train_set)):
        _, label = train_set[index]
        if label not in total_examples:
            total_examples[label] = 1
        else:
            total_examples[label] += 1
    train_indexes = []
    val_indexes = []
    train_indexes_count = {}
    val_indexes_count = {}
    indexes_count = {}
    for index in range(len(train_set)):
        _, label = train_set[index]
        if label not in indexes_count:
            indexes_count[label] = 1
            train_indexes.append(index)
            train_indexes_count[label] = 1
            val_indexes_count[label] = 0
        else:
            indexes_count[label] += 1
            if indexes_count[label] <= int(total_examples[label] * args.train_set_split):
                train_indexes.append(index)
                train_indexes_count[label] += 1
            else:
                val_indexes.append(index)
                val_indexes_count[label] += 1
    print("TRAIN SET INDEXES TOTALS:", train_indexes_count)
    print("VALID SET INDEXES TOTALS:", val_indexes_count)
    train_sampler = SubsetRandomSampler(train_indexes)
    val_sampler = SubsetRandomSampler(val_indexes)

    # Create loaders...
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers,
                                               pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers,
                                             pin_memory=True, sampler=val_sampler)

    print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("TRAINSET LOADER SIZE: ====>>>> ", len(train_loader.sampler))
    print("VALIDSET LOADER SIZE: ====>>>> ", len(val_loader.sampler))
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    # Dataset created...
    print("\nDATASET:", args.dataset)

    # create model
    model = create_model()
    print("\nMODEL:", model)

    best_train_acc1 = 0
    best_val_acc1 = 0

    if args.train:
        ###################
        # Training...
        ###################

        # define loss function (criterion)...
        criterion = nn.CrossEntropyLoss().cuda()

        # define optimizer...
        optimizer = torch.optim.SGD(model.parameters(), lr=args.original_learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)

        # define scheduler...
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.2, verbose=True)

        print("\n################ TRAINING ################")
        best_model_file_path = os.path.join(execution_path, 'best_model.pth.tar')
        best_train_acc1, best_val_acc1 = train_val(train_loader, val_loader, model, criterion, optimizer,
                                                   scheduler, args.epochs, writer, best_model_file_path)

        # save to json file
        writer.export_scalars_to_json(os.path.join(execution_path, 'log.json'))

    if args.extract_logits:
        ######################
        # Extracting logits...
        ######################

        # Train dataset uses val transform to extract...
        train_set = ImageFolder(train_path, transform=inference_transform, selected_classes=args.normal_classes)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers,
                                                   pin_memory=True, sampler=train_sampler)

        val_set = ImageFolder(train_path, transform=inference_transform, selected_classes=args.normal_classes)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers,
                                                 pin_memory=True, sampler=val_sampler)

        test_set = ImageFolder(test_path, transform=inference_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers,
                                                  pin_memory=True, shuffle=True)

        print("\n################ EXTRACTING LOGITS ################")
        best_model_file_path = os.path.join(execution_path, 'best_model.pth.tar')
        extract_logits_from_file(best_model_file_path, model, args.number_of_model_classes, execution_path,
                                 train_loader, val_loader, test_loader, "best_model")

    return best_val_acc1, best_train_acc1


def create_model():
    if args.dataset in ["mnist", "cifar10", "cifar100"]:
        # arch = args.local_model
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=args.number_of_model_classes)
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    else:  # args.dataset == "imagenet2012":
        # arch = args.remote_model
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=args.number_of_model_classes)
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    return model


def extract_logits_from_file(model_file, model, number_of_classes, path,
                             train_loader, val_loader, test_loader, suffix):

    if os.path.isfile(model_file):
        print("\n=> loading checkpoint '{}'".format(model_file))
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(model_file, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        return

    logits_train_file = '{}/{}/{}.pth'.format(path, 'logits', suffix + '_train')
    logits_val_file = '{}/{}/{}.pth'.format(path, 'logits', suffix + '_val')
    logits_test_file = '{}/{}/{}.pth'.format(path, 'logits', suffix + '_test')

    # logits_train_set = extract_logits(model, number_of_classes, train_loader, logits_train_file)
    extract_logits(model, number_of_classes, train_loader, logits_train_file)
    # logits_val_set = extract_logits(model, number_of_classes, val_loader, logits_val_file)
    extract_logits(model, number_of_classes, val_loader, logits_val_file)
    # logits_test_set = extract_logits(model, number_of_classes, test_loader, logits_test_file)
    extract_logits(model, number_of_classes, test_loader, logits_test_file)


def extract_logits(model, number_of_classes, loader, path):

    # print('\nExtract logits on {}set'.format(loader.dataset.set))
    print('Extract logits on {}'.format(loader.dataset))

    logits = torch.Tensor(len(loader.sampler), number_of_classes)
    # logits = torch.Tensor(len(loader.sampler), len(loader.dataset.classes))
    targets = torch.Tensor(len(loader.sampler))
    print("LOGITS:\t\t", logits.size())
    print("TARGETS:\t", targets.size())

    # switch to evaluate mode
    model.eval()

    for batch_id, batch in enumerate(tqdm(loader)):
        img = batch[0]
        # target = batch[2]
        target = batch[1]
        current_bsize = img.size(0)
        from_ = int(batch_id * loader.batch_size)
        to_ = int(from_ + current_bsize)

        img = img.cuda(async=True)

        input_var = Variable(img, requires_grad=False)
        output = model(input_var)

        logits[from_:to_] = output.data.cpu()
        targets[from_:to_] = target

    os.system('mkdir -p {}'.format(os.path.dirname(path)))
    print('save ' + path)
    torch.save((logits, targets), path)
    print('')
    return logits, targets


def train_val(train_loader, val_loader, model, criterion, optimizer, scheduler,
              total_epochs, writer, best_model_file_path):

    best_train_acc1 = 0  # current_best_train_acc1
    best_val_acc1 = 0  # current_best_val_acc1

    # for epoch in range(start_epoch, end_epoch + 1):
    for epoch in range(1, total_epochs + 1):
        print("\n######## EPOCH:", epoch, "OF", total_epochs, "########")

        # Adjusting learning rate (if not using reduce on plateau)...
        # scheduler.step()

        # Print current learning rate...
        for param_group in optimizer.param_groups:
            print("\nLEARNING RATE:\t", param_group["lr"])

        train_acc1 = train(train_loader, model, criterion, optimizer, epoch, writer)
        val_acc1 = validate(val_loader, model, epoch, writer)

        # remember best acc1...
        best_train_acc1 = max(train_acc1, best_train_acc1)
        is_best = val_acc1 > best_val_acc1
        best_val_acc1 = max(val_acc1, best_val_acc1)

        if is_best:
            print("+NEW BEST+ {0:.3f} IN EPOCH {1}!!! SAVING... {2}\n".format(val_acc1, epoch, best_model_file_path))
            # torch.save({'epoch': epoch, 'arch': args.arch, 'state_dict': model.state_dict(),
            #            'best_val_acc1': best_val_acc1, 'optimizer': optimizer.state_dict()},
            #           best_model_file_path)
            full_state = {'epoch': epoch, 'arch': args.arch, 'model_state_dict': model.state_dict(),
                          'best_val_acc1': best_val_acc1, 'optimizer_state_dict': optimizer.state_dict()}
            torch.save(full_state, best_model_file_path)

        print('$$$$ BEST: {0:.3f}\n'.format(best_val_acc1))

        # Adjusting learning rate (if using reduce on plateau)...
        scheduler.step(val_acc1)

    return best_train_acc1, best_val_acc1


def train(train_loader, model, criterion, optimizer, epoch, writer):
    # Meters...
    train_loss = tnt.meter.AverageValueMeter()
    train_acc = tnt.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
    train_conf = tnt.meter.ConfusionMeter(args.number_of_model_classes, normalized=True)

    # switch to train mode
    model.train()

    # Start timer...
    train_batch_start_time = time.time()

    for batch_index, (input_tensor, target_tensor) in enumerate(train_loader):
        batch_index += 1

        # measure data loading time
        train_data_time = time.time() - train_batch_start_time

        input_var = torch.autograd.Variable(input_tensor)
        target_tensor = target_tensor.cuda(async=True)
        target_var = torch.autograd.Variable(target_tensor)

        # compute output
        output_var = model(input_var)

        # print("OUTPUT VAR:", output_var[:])
        # print("TARGET VAR:", target_var[:])

        # compute loss
        loss_var = criterion(output_var, target_var)

        # accumulate metrics over epoch
        train_loss.add(loss_var.data[0])
        train_acc.add(output_var.data, target_var.data)
        train_conf.add(output_var.data, target_var.data)

        # zero grads, compute gradients and do optimizer step
        optimizer.zero_grad()
        loss_var.backward()
        optimizer.step()

        # measure elapsed time
        train_batch_time = time.time() - train_batch_start_time

        if batch_index % args.print_freq == 0:
            print('Train Epoch: [{0}][{1}/{2}]\t'
                  'Data {train_data_time:.6f}\t'
                  'Time {train_batch_time:.6f}\t'
                  'Loss {loss:.4f}\t'
                  'Acc1 {acc1_meter:.2f}\t'
                  'Acc5 {acc5_meter:.2f}'
                  .format(epoch, batch_index, len(train_loader),
                          train_data_time=train_data_time,
                          train_batch_time=train_batch_time,
                          loss=train_loss.value()[0],
                          acc1_meter=train_acc.value()[0],
                          acc5_meter=train_acc.value()[1],
                          )
                  )

        # Restart timer...
        train_batch_start_time = time.time()

    print("\nEPOCH TRAIN RESULTS")
    print("LOSS:\t\t", train_loss.value())
    print("ACCURACY:\t", train_acc.value())
    print("\nCONFUSION:\n", train_conf.value())

    writer.add_scalar('train/loss', train_loss.value()[0], epoch)
    writer.add_scalar('train/acc1', train_acc.value()[0], epoch)
    writer.add_scalar('train/acc5', train_acc.value()[1], epoch)

    print('\n#### TRAIN: {acc1:.3f}\n\n'.format(acc1=train_acc.value()[0]))
    # confusion = dict(np.ndenumerate(train_conf.value()))
    # confusion = {str(key):float(value) for key,value in confusion.items()}
    # writer.add_scalars('train/confusion', confusion, epoch)

    return train_acc.value()[0]


def validate(val_loader, model, epoch, writer):
    # Meters...
    val_acc = tnt.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
    val_conf = tnt.meter.ConfusionMeter(args.number_of_model_classes, normalized=True)

    # switch to evaluate mode
    model.eval()

    # Start timer...
    val_batch_start_time = time.time()

    for batch_index, (input_tensor, target_tensor) in enumerate(val_loader):
        batch_index += 1

        # measure data loading time
        val_data_time = time.time()-val_batch_start_time

        input_var = torch.autograd.Variable(input_tensor, volatile=True)
        target_tensor = target_tensor.cuda(async=True)
        target_var = torch.autograd.Variable(target_tensor, volatile=True)

        # compute output
        output_var = model(input_var)

        # accumulate metrics over epoch
        val_acc.add(output_var.data, target_var.data)
        val_conf.add(output_var.data, target_var.data)

        # measure elapsed time
        val_batch_time = time.time()-val_batch_start_time

        if batch_index % args.print_freq == 0:
            print('Val Epoch: [{0}][{1}/{2}]\t'
                  'Data {val_data_time:.6f}\t'
                  'Time {val_batch_time:.6f}\t'
                  'Acc1 {acc1_meter:.2f}\t'
                  'Acc5 {acc5_meter:.2f}'
                  .format(epoch, batch_index, len(val_loader),
                          val_data_time=val_data_time,
                          val_batch_time=val_batch_time,
                          acc1_meter=val_acc.value()[0],
                          acc5_meter=val_acc.value()[1],
                          )
                  )

        # Restart timer...
        val_batch_start_time = time.time()

    print("\nEPOCH VAL RESULTS")
    print("ACCURACY:\t", val_acc.value())
    print("\nCONFUSION:\n", val_conf.value())

    writer.add_scalar('val/acc1', val_acc.value()[0], epoch)
    writer.add_scalar('val/acc5', val_acc.value()[1], epoch)

    print('\n####  VAL: {acc1:.3f}\n'.format(acc1=val_acc.value()[0]))
    # confusion = dict(np.ndenumerate(val_conf.value()))
    # confusion = {str(key):float(value) for key,value in confusion.items()}
    # writer.add_scalars('val/confusion', confusion, epoch)

    return val_acc.value()[0]


def main():

    if not (args.train or args.extract_logits):
        print("\nNOTHING TO DO!!!\n")
        sys.exit()

    overall_statistics = {}

    for experiment in args.experiments:

        print("\n**************** EXPERIMENT:", experiment.upper(), "****************")

        car_results = {}
        car_statistics = pd.DataFrame()

        if args.local_model is not None:
            args.arch = args.local_model
        else:
            args.arch = args.remote_model

        experiment_path = os.path.join("artifacts", args.dataset, args.arch, experiment)
        print("\nEXPERIMENT PATH:", experiment_path)

        experiment_configs = experiment.split("+")

        for config in experiment_configs:
            config = config.split("~")
            if config[0] == "nmc":
                args.number_of_model_classes = int(config[1])
                print("NUMBER OF MODEL CLASSES:", args.number_of_model_classes)

        for execution in range(1, args.executions + 1):

            # Using seeds
            random.seed(execution)
            numpy.random.seed(execution)
            torch.manual_seed(execution)
            torch.cuda.manual_seed(execution)

            # Preparing paths...
            execution_path = os.path.join(experiment_path, "exec" + str(execution))
            if not os.path.exists(execution_path):
                os.makedirs(execution_path)

            # Preparing logger...
            writer = SummaryWriter(log_dir=execution_path)
            writer.add_text(str(vars(args)), str(vars(args)))

            # Printing args...
            # print("\nARGUMENTS: ", vars(args))

            print("\n################ EXECUTION:", execution, "OF", args.executions, "################")

            best_val_acc1, best_train_acc1 = execute(execution_path, writer)

            # Results and auc_statistics...
            car_results["BEST_VAL_ACC1"], car_results["BEST_TRAIN_ACC1"] = best_val_acc1, best_train_acc1
            car_statistics = car_statistics.append(car_results, ignore_index=True)
            car_statistics = car_statistics[["BEST_VAL_ACC1", "BEST_TRAIN_ACC1"]]

        print("\n################################\n", "EXPERIMENT RESULTS", "\n################################\n")
        print("\n", experiment.upper())
        print("\n", car_statistics.transpose())
        print("\n", car_statistics.describe())
        overall_statistics[experiment] = car_statistics

    print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n", "OVERALL RESULTS", "\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
    for key in overall_statistics:
        print("\n", key.upper())
        print("\n", overall_statistics[key].transpose())
        print("\n", overall_statistics[key].describe())


if __name__ == '__main__':
    main()
