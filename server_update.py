import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset,  match_loss, get_time, \
    TensorDataset, epoch, DiffAugment, ParamDiffAug

import logging
import copy
import random
from reparam_module import ReparamModule
import torch.utils.data
import warnings
import gc

from glad_utils import *

import time


warnings.filterwarnings("ignore", category=DeprecationWarning)


def server_update(args, latents, f_latents, starting_params, target_params, G, testloader, channel, im_size, num_classes, round):
    # starting_parms 是起始参数，target_params 是目标参数
    # 设置评估迭代次数
    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    # 获取模型评估池
    model_eval_pool = get_eval_pool('my', args.model, args.model)
    # 记录所有实验的性能
    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []
    # 保存中间结果
    data_save = []

    if args.max_experts is not None and args.max_files is not None:
        # 根据最大专家数和文件数计算总专家数
        args.total_experts = args.max_experts * args.max_files


    if args.batch_syn is None:
        # 如果没有指定批量合成图像大小，则根据类别数和 ipc 计算
        args.batch_syn = num_classes * args.ipc
    # 判断是否需要分布式训练
    args.distributed = len(args.gpu) > 1
    # 生成合成图像的标签
    label_syn = torch.tensor([np.ones(args.ipc) * i for i in range(num_classes)], dtype=torch.long, requires_grad=False,
                             device=args.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]
    # 初始化学习率 syn_lr
    syn_lr = torch.tensor(args.lr_teacher, requires_grad=True).to(args.device)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    # 初始化 学习率优化器 optimizer_lr, 使用 SGD 优化器，学习率初始化为 args.lr_lr, 动量 0.5
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    # 获取图像优化器 optimizer_img, 使用 get_optimizer_img 函数获取,用于优化生成图像的潜在向量，即GAN中的潜在空间参数
    optimizer_img = get_optimizer_img(latents=latents, f_latents=f_latents, G=G, args=args)
    # 定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

    for it in range(0, args.Iteration+1):
        student_net = get_network(args.model, channel, num_classes, im_size, width=args.width, depth=args.depth, dist=False, args=args).to(args.device)  # get a random model
        # 计算学生网络中总参数量
        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])
        # 将 target_params 中每个参数展平并拼接成一个一维张量
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)
        # 将 start_params 中每个参数展平并拼接成一个一维张量， 并设置为可求导
        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
        # 将 start_params 中每个参数展平并拼接成一个一维张量
        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)
        # 将 student_net 包装成一个 ReparamModule 类的实例，ReparamModule 是一个自定义的模块，用于重新参数化学生网络中的参数，以便在后续的训练过程中更方便地操作和更新这些参数。
        student_net = ReparamModule(student_net)
        # 初始化一个与 starting_params 形状相同且初始值为零的张量 gradient_sum，并将其移动到指定设备（如GPU）上。该张量用于累加梯度，且不需要计算梯度
        gradient_sum = torch.zeros(starting_params.shape).requires_grad_(False).to(args.device)
        # 初始化 param_dist 为 0
        param_dist = torch.tensor(0.0).to(args.device)
        # 使用 torch.nn.functional.mse_loss 计算 starting_params 和 target_params 的均方差，并将结果累加到 param_dist 上
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
        # 判断是否需要进行分布式训练
        if args.distributed:
            student_net = torch.nn.parallel.DistributedDataParallel(student_net, device_ids=[args.local_rank])

        student_net.train()
        # 从潜在空间生成合成图像，并根据参数设置进行不同的处理
        # 初始化 syn_images 为 latents 的副本
        syn_images = latents[:]
        if args.space == "wp":
            if args.layer != -1:
                # 使用 latent_to_im 将潜在向量转换为图像，并拼接结果
                with torch.no_grad():
                    syn_images = torch.cat([latent_to_im(G, (syn_image_split.detach(), f_latents_split.detach()),
                                                        args=args).detach() for
                                           syn_image_split, f_latents_split, label_syn_split in
                                           zip(torch.split(syn_images, args.sg_batch),
                                               torch.split(f_latents, args.sg_batch),
                                               torch.split(label_syn, args.sg_batch))])
            else:
                # 直接将潜在向量转换为图像并拼接结果
                with torch.no_grad():
                    syn_images = torch.cat([latent_to_im(G, (syn_image_split.detach(), None), args=args).detach() for
                                           syn_image_split, label_syn_split in
                                           zip(torch.split(syn_images, args.sg_batch),
                                               torch.split(label_syn, args.sg_batch))])
            # 设置 syn_images 为可求导
            syn_images.requires_grad_(True)
        # 将 syn_images 转换为不可求导的张量 image_syn
        image_syn = syn_images.detach()
        # 初始化合成图像的标签
        y_hat = label_syn
        # 创建空列表用于存储后续计算中的数据和索引
        x_list = []
        y_list = []
        indices_chunks = []
        indices_chunks_copy = []
        original_x_list = []
        gc.collect()
        # 初始化两个梯度变量，用于存储标签和图像的梯度
        syn_label_grad = torch.zeros(label_syn.shape).to(args.device).requires_grad_(False)
        syn_images_grad = torch.zeros(syn_images.shape).requires_grad_(False).to(args.device)
        # 通过合成图像和标签进行训练，更新学生网络的参数
        for il in range(args.syn_steps):
            # 初始化索引列表，如果indices_chunks为空，则生成随机打乱的索引
            if not indices_chunks:
                indices = torch.randperm(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))
            # 随机打乱并分批处理合成图像和标签
            these_indices = indices_chunks.pop()
            indices_chunks_copy.append(these_indices.clone())
            # 从syn_images和y_hat中取出对应的数据和标签
            x = syn_images[these_indices]
            this_y = y_hat[these_indices]
            # 保存原始数据
            original_x_list.append(x)
            # 数据增强
            x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)
            # 保存增强后的数据
            x_list.append(x.clone())
            y_list.append(this_y.clone())
            # 从 student_params 中获取最新的参数
            forward_params = student_params[-1]
            forward_params = copy.deepcopy(forward_params.detach()).requires_grad_(True)
            # 如果是分布式训练，扩展参数；否则直接使用
            if args.distributed:
                forward_params_expanded = forward_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params_expanded = forward_params
            # 使用当前学生网络参数计算输出
            x = student_net(x, flat_param=forward_params_expanded)
            # 计算交叉熵损失
            ce_loss = criterion(x, this_y)
            # 计算梯度
            grad = torch.autograd.grad(ce_loss, forward_params, create_graph=True, retain_graph=True)[0]
            # 更新学生网络参数
            student_params.append(forward_params - syn_lr.item() * grad.detach().clone())
            # 将梯度累加到 gradient_sum
            gradient_sum = gradient_sum + grad.detach().clone()

        for il in range(args.syn_steps):
            # 从 student_params 中获取当前参数
            w = student_params[il]
            # 如果是分布式训练，将参数 w 扩展为多设备使用的格式
            if args.distributed:
                w_expanded = w.unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                w_expanded = w
            # 用 student_net 和当前参数 w_expanded 计算输出
            output = student_net(x_list[il], flat_param=w_expanded)
            # 计算交叉熵损失
            if args.batch_syn:
                ce_loss = criterion(output, y_list[il])
            else:
                ce_loss = criterion(output, y_hat)
            # 计算梯度
            grad = torch.autograd.grad(ce_loss, w, create_graph=True, retain_graph=True)[0]

            # Square term gradients.
            # 计算 平方项，即学习率的平方乘以梯度的内积
            square_term = syn_lr.item() ** 2 * (grad @ grad)
            # 计算单个项，设计梯度，累计梯度，起始参数 和目标参数的线性组合
            single_term = 2 * syn_lr.item() * grad @ (
                        syn_lr.item() * (gradient_sum - grad.detach().clone()) - starting_params + target_params)
            # 将平方项和单个项相加后除以参数距离
            per_batch_loss = (square_term + single_term) / param_dist
            # 计算原始输入图像的梯度
            gradients = torch.autograd.grad(per_batch_loss, original_x_list[il], retain_graph=False)[0]
            # 将这些梯度累加到 syn_images_grad 的对应位置
            with torch.no_grad():
                syn_images_grad[indices_chunks_copy[il]] += gradients

        # ---------end of computing input image gradients and learning rates--------------

        del w, output, ce_loss, grad, square_term, single_term, per_batch_loss, gradients, student_net, w_expanded, forward_params, forward_params_expanded
        # 将优化器的梯度清零
        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()
        # 将 syn_lr 设置为需要计算梯度的张量
        syn_lr.requires_grad_(True)
        # 起始参数 减去 学习率乘以梯度累加和 再减去 目标参数，衡量了当前参数与目标参数之间的差距
        grand_loss = starting_params - syn_lr * gradient_sum - target_params
        # 计算L2范数的平方
        grand_loss = grand_loss.dot(grand_loss)
        grand_loss = grand_loss / param_dist
        # 计算 grand_loss 对 syn_lr 的梯度，将其赋值给 syn_lr.grad
        lr_grad = torch.autograd.grad(grand_loss, syn_lr)[0]
        syn_lr.grad = lr_grad
        # 使用 optimizer_lr 对 syn_lr 进行优化
        optimizer_lr.step()
        # 清零 optimizer_lr 的梯度
        optimizer_lr.zero_grad()
        # 将 image_syn 设置为需要计算梯度的张量，并将 syn_images_grad 作为其梯度
        image_syn.requires_grad_(True)
        image_syn.grad = syn_images_grad.detach().clone()

        del syn_images_grad
        del lr_grad

        for _ in student_params:
            del _
        for _ in x_list:
            del _
        for _ in y_list:
            del _

        torch.cuda.empty_cache()

        gc.collect()

        if args.space == "wp":
            # 调用 gan_backward 方法，将梯度反向传播到 latents 和 f_latents
            # this method works in-line and back-props gradients to latents and f_latents
            gan_backward(latents=latents, f_latents=f_latents, image_syn=image_syn, G=G, args=args)

        else:
            # 直接将 image_syn 的梯度赋值给 latents 的梯度
            latents.grad = image_syn.grad.detach().clone()

        optimizer_img.step()
        optimizer_img.zero_grad()


        if it%10 == 0:
            logging.info('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))

        if it == args.Iteration:
            data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
            torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%dipc.pt'%(args.dataset, args.model, args.ipc)))

    image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())

    return image_syn_train, label_syn_train


if __name__ == '__main__':
    import shared_args

    parser = shared_args.add_shared_args()
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_syn', type=int, default=None, help='batch size for syn data')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--load_all', action='store_true')
    parser.add_argument('--max_start_epoch', type=int, default=5)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--max_experts', type=int, default=None)
    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')

    parser.add_argument('--lr_img', type=float, default=10000, help='learning rate for pixels or f_latents')
    parser.add_argument('--lr_w', type=float, default=10, help='learning rate for updating synthetic latent w')
    parser.add_argument('--lr_lr', type=float, default=1e-06, help='learning rate learning rate')
    parser.add_argument('--lr_g', type=float, default=0.1, help='learning rate for gan weights')

    args = parser.parse_args()


