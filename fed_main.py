import logging
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import sys
import random
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, match_loss, get_time, \
    TensorDataset, epoch, DiffAugment, ParamDiffAug, epoch_global, get_logger
from clients import Client
from server_update import server_update
from glad_utils import *
from baseline_methods import *
# 设置为单进程
import os
os.environ['RANK'] = '0'        # 当前进程序号
os.environ['WORLD_SIZE'] = '1'  # 总进程数
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
# 根据传入的参数，构建一个字符串
def get_fname(args):
    parserString = "fedmain_" + str(args.dataset) + "_Layer" + str(args.layer) + "_Iteration" + str(
        args.Iteration) + "_nworkers" + str(args.nworkers) + \
                   "_beta" + str(args.beta) + "_ipc" + str(args.ipc) + "_round" + str(args.round)
    return parserString


def main(args):
    input_str = ' '.join(sys.argv)
    # input_str = r"E:\FL+DM\FedDG-main\fed_main.py"
    print(input_str)
    # 检查args对象是否有'local_rank'属性，或者该属性的值是否小于0。如果满足任一条件，就将local_rank设置为0
    if not hasattr(args, 'local_rank') or args.local_rank < 0:
        args.local_rank = 0
    print(args.local_rank)
    # 初始化日志记录器，生成包含参数和时间戳的日志文件路径
    logger = get_logger('log/' + get_fname(args) + time.strftime("%Y%m%d-%H%M%S") + ".log",
                        distributed_rank=args.local_rank)
    # 遍历所有命令行参数，按字母顺序记录参数名和值到日志
    for k, v in sorted(vars(args).items()):
        logger.info(str(k) + '=' + str(v))
    # 记录完整的命令行输入字符串到日志
    logger.info(input_str)
    # 初始化分布式训练环境，使用gloo后端创建进程组
    dist.init_process_group(backend='gloo')
    # dist.init_process_group(backend='nccl')
    # 设置当前进程使用的GPU设备并配置设备参数
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda', args.local_rank) if torch.cuda.is_available() else 'cpu'
    # 初始化差异增强参数并根据策略设置增强开关
    args.dsa_param = ParamDiffAug()
    # 根据args.dsa_strategy的值，决定是否启用数据增强(dsa设为True或False)
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    # 设置随机种子，确保实验可复现
    # 初始化PyTorch CPU和GPU随机数生成器
    # 初始化NumPy和Python原生随机种
    # 禁用CUDNN非确定性算法保证计算确定性
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # 检查数据路径是否存在，若不存在则创建
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)
    # 创建运行目录，使用当前时间戳和参数信息作为目录名
    run_dir = "{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), get_fname(args))
    # 创建保存路径，使用当前时间戳和参数信息作为目录名
    args.save_path = os.path.join(args.save_path, "fedgan", run_dir)
    # 如果保存路径不存在，则创建该路径
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    # 调用get_dataset加载数据集并获取图像通道、尺寸、类别数等参数，初始化训练/测试数据集及数据加载器
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(
        args.dataset, args.data_path, args.batch_real, args.res, args=args)
    # 根据评估模式调用get_eval_pool获取待评估的模型列表
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    # 检测可用GPU数量判断是否启用分布式训练模式
    args.distributed = torch.cuda.device_count() > 1

    # 通过Dirichlet分布将训练数据分配给各客户端，返回数据集、标签和类别索引
    each_worker_data, each_worker_label, indices_cl_classes = build_client_dataset_dirichlet(dst_train, class_map,
                                                                                             num_classes, args)
    # 统计总数据量，计算各客户端数据占比（fed_avg_freqs)
    total_data_points = len(dst_train)
    fed_avg_freqs = [len(each_worker_data[each_worker]) / total_data_points for each_worker in
                     range(args.nworkers)]
    # 循环打印各客户端数据量及各分类样本数量，验证数据分布合理性
    for each_client in range(args.nworkers):
        print(each_client, " ", len(each_worker_data[each_client]))
        print([len(indices_cl_classes[each_client][c]) for c in range(10)])
        print("=" * 100)
    print(fed_avg_freqs, sum(fed_avg_freqs))
    # 从指定客户端的特定类别 c 中随机选取 n 图像样本并转为GPU张量
    def get_images(c, n, each_worker):  # get random n images of class c from client each_worker
        # np.random.permutation打乱indices_cl_classes[each_worker][c]的索引，取前n个，然后从each_worker_data中取出这些图像，并转换到cuda上
        idx_shuffle = np.random.permutation(indices_cl_classes[each_worker][c])[:n]
        return each_worker_data[each_worker][idx_shuffle].cuda(non_blocking=True)
    # global_model_acc和global_model_loss的初始化,这两个字典用来记录不同模型在实验中的准确率和损失
    global_model_acc = dict()  # record performances of all experiments
    global_model_loss = dict()
    # 首先初始化'Origin'键的空列表，然后遍历model_eval_pool中的每个键，都添加到这两个字典里
    global_model_acc['Origin'] = []
    global_model_loss['Origin'] = []
    for key in model_eval_pool:
        global_model_acc[key] = []
        global_model_loss[key] = []
    """initialize clients"""
    # 创建 Clients 列表
    Clients = []
    # 循环遍历每个客户端
    for each_worker in range(args.nworkers):
        # 根据模型参数创建神经网络架构
        net = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width, args=args).to(
            args.device)  # get a random model
        # 使用分布式数据并行包装模型支持多GPU训练
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])
        net.train()
        # 初始化SGD优化器和交叉熵损失函数
        optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)
        criterion = nn.CrossEntropyLoss().cuda()
        # 创建Client实例并传入数据、模型及相关参数
        client = Client(each_worker, optimizer=optimizer_net, criterion=criterion,
                        each_worker_data=each_worker_data[each_worker],
                        each_worker_label=each_worker_label[each_worker],
                        indices_cl_class=indices_cl_classes[each_worker],
                        model=net, args=args)
        # 将客户端实例添加到客户端列表
        Clients.append(client)

    # exit(0)
    """initialize global model"""
    # 根据不同的空间类型（比如生成对抗网络中的潜在空间类型）来初始化生成器G和相关参数。'p'指普通潜在空间，而'wp'指StyleGAN中的W+空间
    if args.space == 'p':
        G, zdim = None, None
    elif args.space == 'wp':
        # 调用 load_sgxl函数来初始化生成器G，并获取其他相关参数，如潜在空间维度、W+空间维度和图像类别数。
        G, zdim, w_dim, num_ws = load_sgxl(args.res, args)
    else:
        exit("unknown space: %s" % args.space)
    # 创建全局模型 net，用 get_network 函数根据参数生成模型结构，并将其转移到设备
    net = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width, args=args).to(
        args.device)  # get a random model
    # 使用DistributedDataParallel进行分布式训练包装
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])
    net.train()
    # 设置优化器为SGD，学习率由args.lr_net指定，并初始化优化器的梯度为零，定义交叉熵损失函数
    optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
    optimizer_net.zero_grad()
    criterion = nn.CrossEntropyLoss().cuda()

    # 初始化测试模型列表和优化器列表
    test_nets = []
    test_optimizers = []
    # 遍历待评估模型池中的每个模型名称
    for real_model in model_eval_pool:
        # 调用 get_network 函数，为每个模型名称创建对应网络结构实例
        real_net = get_network(real_model, channel, num_classes, im_size, depth=args.depth, width=args.width,
                               args=args).to(
            args.device)  # get a random model
        # 将模型包装为分布式并行模型
        real_net = torch.nn.parallel.DistributedDataParallel(real_net, device_ids=[args.local_rank])
        real_net.train()
        # 配置SGD优化器参数并初始化优化器
        optimizer_real_net = torch.optim.SGD(filter(lambda p: p.requires_grad, real_net.parameters()), lr=args.lr_net,
                                        momentum=0.9, weight_decay=args.reg)
        # 将优化器的梯度设置为零
        optimizer_real_net.zero_grad()
        # 将模型和优化器存入对应列表
        test_nets.append(real_net)
        test_optimizers.append(optimizer_real_net)
    # 打印模型评估池内容
    print(model_eval_pool)
    # 检查是否需要恢复训练（args.recover为真时触发）
    if args.recover:
        save_dir = os.path.join(args.saved_model_dir, "local_model")
        net.load_state_dict(torch.load(os.path.join(save_dir, "model_{0:05d}.pt".format(args.recover_round))))
        net = net.to(args.device)
        net.eval()
        for net_name, real_net, optimizer_real_net in zip(model_eval_pool, test_nets, test_optimizers):
            save_dir = os.path.join(args.saved_model_dir, net_name)
            real_net.load_state_dict(torch.load(os.path.join(save_dir, "model_{0:05d}.pt".format(args.recover_round))))
            real_net = real_net.to(args.device)
            real_net.eval()

    """federated training"""
    for i in range(args.round):
        logger.info("=" * 50 + " Round: " + str(i) + " " + "=" * 50)
        # 初始化image_syn_trains和label_syn_trains，用于存储合成的图像和标签
        image_syn_trains = []
        label_syn_trains = []
        # 将全局模型参数复制到CPU上，并保存到global_para变量中，以便客户端使用
        global_para = net.cpu().state_dict()
        # 确定选择参与本轮的客户端数量m，使用args.frac乘以总客户端数，但至少选1个
        m = max(int(args.frac * args.nworkers), 1)
        # 从所有客户端中随机选择m个客户端，并记录它们的索引
        idxs_users = np.random.choice(range(args.nworkers), m, replace=False)
        logging.info('\nChoosing users {}'.format(' '.join(map(str, idxs_users))))
        # 如果处于恢复模式且当前轮次小于恢复轮次，则跳过本轮
        if args.recover:
            if i <= args.recover_round:
                continue
        # 初始化起始参数列表和目标参数列表
        starting_params_list = []
        target_params_list = []
        #  遍历选中的客户端，加载全局模型参数到客户端模型，然后进行本地训练，获取起始和目标参数，并存入列表
        for each_worker in idxs_users:
            Clients[each_worker].model.load_state_dict(global_para)
            loss, starting_params, target_params, _ = Clients[each_worker].train_net(local_round=args.Iteration_g,
                                                                                      batch_size=args.batch_train,
                                                                                      device=args.device,
                                                                                      is_mtt=True, args=args)
            starting_params_list.append(starting_params)
            target_params_list.append(target_params)

        for each_worker in idxs_users:
            logger.info(input_str)
            print('\n')
            logging.info('\nChoosing users {}'.format(' '.join(map(str, idxs_users))))
            logger.info("=" * 50 + " Client: " + str(each_worker) + " " + "=" * 50)
            latents, f_latents, label_syn = prepare_latents(channel=channel, num_classes=num_classes,
                                                            im_size=im_size, zdim=zdim, G=G,
                                                            class_map_inv=class_map_inv,
                                                            get_images=get_images, each_worker=each_worker,
                                                            args=args)
            image_syn_train, label_syn_train = server_update(args, latents, f_latents,
                                                             starting_params_list[each_worker],
                                                             target_params_list[each_worker], G,
                                                             testloader, channel, im_size, num_classes, i)
            image_syn_trains.append(image_syn_train)
            label_syn_trains.append(label_syn_train)

            if ((i + 1) % 5 == 0 and not args.not_save_file) or (i + 1 == args.round) or args.save_all:
                logging.info('=' * 50 + 'Saving' + '=' * 50)
                save_dir = os.path.join(args.logdir, args.dataset, args.save_path, str(each_worker))

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                torch.save(image_syn_train.cpu(), os.path.join(save_dir, "images_{0:05d}.pt".format(i)))
                torch.save(label_syn_train.cpu(), os.path.join(save_dir, "labels_{0:05d}.pt".format(i)))

        image_syn_trains = torch.cat(image_syn_trains)
        label_syn_trains_ = torch.cat(label_syn_trains)
        dst_syn_train = TensorDataset(image_syn_trains, label_syn_trains_)
        trainsampler = torch.utils.data.distributed.DistributedSampler(dst_syn_train)
        trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, sampler=trainsampler,
                                                  num_workers=0)

        for il in range(args.inner_loop):
            loss_avg, acc_avg = epoch_global('train', trainloader, net, optimizer_net, criterion,
                                             args, aug=True if args.dsa else False)
            if il % 100 == 0:
                logger.info(
                    '%s Evaluate_%s_%02d: train loss = %.6f train acc = %.4f' % (
                    args.model, get_time(), il, loss_avg, acc_avg))
            for net_name, real_net, optimizer_real_net in zip(model_eval_pool, test_nets, test_optimizers):
                loss_avg, acc_avg = epoch_global('train', trainloader, real_net, optimizer_real_net, criterion,
                                                 args, aug=True if args.dsa else False)
                if il % 100 == 0:
                    logger.info(
                        '%s Evaluate_%s_%02d: train loss = %.6f train acc = %.4f' % (
                        net_name, get_time(), il, loss_avg, acc_avg))
        with torch.no_grad():
            if (i + 1) % 5 == 0:
                logging.info('=' * 50 + 'Saving model' + '=' * 50)
                save_dir = os.path.join(args.logdir, args.dataset, args.save_path, "local_model")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(net.cpu().state_dict(), os.path.join(save_dir, "model_{0:05d}.pt".format(i)))
            loss_avg, acc_avg = epoch_global('test', testloader, net, optimizer_net, criterion, args,
                                             aug=True if args.dsa else False)
            global_model_acc['Origin'].append(acc_avg)
            global_model_loss['Origin'].append(loss_avg)
            logger.info('%s Evaluate_%s: val loss = %.6f val acc = %.4f' % (args.model, get_time(), loss_avg, acc_avg))

            for net_name, real_net, optimizer_real_net in zip(model_eval_pool, test_nets, test_optimizers):
                if (i + 1) % 5 == 0:
                    save_dir = os.path.join(args.logdir, args.dataset, args.save_path, net_name)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(real_net.cpu().state_dict(), os.path.join(save_dir, "model_{0:05d}.pt".format(i)))
                loss_avg, acc_avg = epoch_global('test', testloader, real_net, optimizer_real_net, criterion, args,
                                                 aug=True if args.dsa else False)
                global_model_acc[net_name].append(acc_avg)
                global_model_loss[net_name].append(loss_avg)
                logger.info(
                    '%s Evaluate_%s: val loss = %.6f val acc = %.4f' % (net_name, get_time(), loss_avg, acc_avg))

    logger.info("Test Loss: " + str(global_model_loss))
    logger.info("Test Acc: " + str(global_model_acc))
    logger.info(input_str)


if __name__ == '__main__':
    import shared_args

    parser = shared_args.add_shared_args()

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for pixels or f_latents')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate learning rate')
    parser.add_argument('--lr_w', type=float, default=1, help='learning rate for updating synthetic latent w')
    parser.add_argument('--lr_g', type=float, default=0.001, help='learning rate for gan weights')

    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--inner_loop', type=int, default=1000, help='inner loop')
    parser.add_argument('--Iteration_g', type=int, default=10, help='inner loop')
    parser.add_argument('--outer_loop', type=int, default=1, help='outer loop')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_syn', type=int, default=None, help='batch size for syn data')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--load_all', action='store_true')
    parser.add_argument('--max_start_epoch', type=int, default=5)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--max_experts', type=int, default=None)
    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    # 添加的代码
    parser.add_argument('--learn_g_after', type=int, default=0, help='Start learning generator after X iterations')
    args = parser.parse_args()
    main(args)
