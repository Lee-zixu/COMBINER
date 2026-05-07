import os 
import sys
import argparse
import logging
import warnings 
import time 
import itertools
import random

import numpy as np 
import torch 
import torch.optim as optim 
from torch.autograd import Variable
from torch.utils.data import dataloader
from tqdm import tqdm
# from tensorboardX import SummaryWriter

import utils
import datasets_openclip as datasets
import model as img_text_model
import test

from torch.cuda.amp import autocast as autocast, GradScaler

import setproctitle
proc_title = "python-c"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
setproctitle.setproctitle(proc_title)  
warnings.filterwarnings("ignore")
torch.set_num_threads(2)

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
parser.add_argument('--dataset', default = 'fashioniq', help = "data set type")
parser.add_argument('--fashioniq_path', default = "./data/CIR_data/fashion_iq_data/")
parser.add_argument('--shoes_path', default = "./data/CIR_data/shoes_data/")
parser.add_argument('--cirr_path', default = "./data/CIR_data/cirr_data/CIRR/")

parser.add_argument('--optimizer', default = 'adamw')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=42)   
parser.add_argument('--lr', type=float, default=1e-4) 
parser.add_argument('--clip_lr', type=float, default=1e-5) 
parser.add_argument('--img_encoder', type=str, default='ViT-B/16')
parser.add_argument('--lr_decay', type=int, default=5)
parser.add_argument('--lr_div', type=float, default=0.1)  
parser.add_argument('--clip_lr_div', type=float, default=0.1)  


parser.add_argument('--max_decay_epoch', type=int, default=10) 
parser.add_argument('--feature_dim', type=int, default=1024)

parser.add_argument('--lambda_', type=float, default=1.0) 
parser.add_argument('--eta_', type=float, default=1.0) 
parser.add_argument('--mu_', type=float, default=0.1)
parser.add_argument('--nu_', type=float, default=10)
parser.add_argument('--kappa_', type=float, default=0.5)
parser.add_argument('--tau_', type=float, default=0.1)
parser.add_argument('--P', type=int, default=4)
parser.add_argument('--Q', type=int, default=8)
parser.add_argument('--recon_w', type=float, default=1)
parser.add_argument('--aug_num', type=int, default=0)
parser.add_argument('--cutoff_length', type=int, default=5)

parser.add_argument('--pro_w', type=float, default=1.0)
parser.add_argument('--kl_w', type=float, default=0.2)

parser.add_argument('--num_cluster', type=int, default=700)
parser.add_argument('--temperature_cluster', type=float, default=0.2)

parser.add_argument('--model_dir', default='./experiment',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--save_summary_steps', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--node', type=str, default='')
args = parser.parse_args()
if args.dataset == "fashion200k":
    torch.multiprocessing.set_sharing_strategy('file_system')
args.num_cluster = [args.num_cluster]
print(args.num_cluster)



def load_dataset():
    """Loads the input datasets."""
    print('Reading dataset ', args.dataset)
    import open_clip
    clip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-H-14', pretrained=os.path.join('./Pretrain_Model/CLIP-ViT-H-14-laion2B-s32B-b79K', 'open_clip_pytorch_model.bin'))
    if args.dataset == 'fashioniq':
        trainset = datasets.FashionIQ(
            path = args.fashioniq_path,
            transform=[preprocess_train, preprocess_val])
    elif args.dataset == 'shoes':
        trainset = datasets.Shoes(
            path = args.shoes_path,
            transform=[preprocess_train, preprocess_val])
    elif args.dataset == 'cirr':
        trainset = datasets.CIRR(
            path = args.cirr_path,
            transform = [preprocess_train, preprocess_val],
            case_look=False
        )
   
    else:
        print('Invalid dataset', args.dataset)
        sys.exit()

    print('trainset size:', len(trainset))

    return trainset

def set_bn_eval(m): 
    classname = m.__class__.__name__ 
    if classname.find('BatchNorm2d') != -1: 
        m.eval() 

def create_model_and_optimizer():
    model = img_text_model.COMBINER(hidden_dim=args.feature_dim, dropout=args.dropout_rate, local_token_num=args.Q, global_token_num = args.P, t = args.tau_)
    model.cuda()

    params = list(model.named_parameters())
    param_group = [
        {'params': [p for n, p in params if any(nd in n for nd in ['clip'])], 'lr': args.clip_lr, "name": "clip"},
        {'params': [p for n, p in params if not any(nd in n for nd in ['clip'])], 'lr': args.lr, "name": "no clip"},
    ]
    optimizer = torch.optim.AdamW(param_group, lr=args.lr, weight_decay = args.weight_decay)

    return model, optimizer

import faiss

@torch.no_grad()
def compute_features(travelloader, model, args):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(travelloader.dataset), 1024)
    with tqdm(total=len(travelloader), desc="Inference dataset for clustering... ") as t:
        with torch.no_grad():
            front = 0
            for data in travelloader:
                img2 = data['target_img_data'].cuda()
                emb_img2 = model.extract_retrieval_target(img2).cpu()
                features[front:(front+data['target_img_data'].shape[0])] = emb_img2.cpu()
                front = front+data['target_img_data'].shape[0]
                t.update()

    return features.cpu()


@torch.no_grad()
def run_kmeans_cpu(x, args):
    """
    Args:
        x: data to be clustered
    """

    print('performing kmeans clustering')
    results = {'im2cluster': [], 'centroids': [], 'density': [], 'centroids_img': []}

    for seed, num_cluster in enumerate(args.num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Kmeans(d, k)
        
        clus.verbose = True
        clus.verbose = False
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 500
        clus.min_points_per_centroid = 5
        clus.index = faiss.IndexFlatIP(d)
        print("Begin Train")
        st = time.time()
        clus.train(x)
        et = time.time()
        logging.info(f"Cluster{num_cluster} train time: {(et - st):.3f}ms")
        D, I = clus.index.search(x, 1) 
        im2cluster = [int(n[0]) for n in I]

        centroids = clus.centroids.reshape(k, d)
        Dcluster = [[] for c in range(k)]
        for im, i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])

        density = np.zeros(k)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d

        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10),
                               np.percentile(density, 90)) 
        density = args.temperature_cluster * density / density.mean() 

        centroids = torch.Tensor(centroids).cuda()

        im2cluster = torch.LongTensor(im2cluster).cuda()
        density = torch.Tensor(density).cuda()

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)

    return results

def train(model, optimizer, dataloader, scaler, epoch, cluster_result):
    model.train()
    model.apply(set_bn_eval)
    summ = []
    loss_avg = utils.RunningAverage()
    with tqdm(total=len(dataloader)) as t:
        front = 0
        for i, data in enumerate(dataloader):
            img1 = data['source_img_data'].cuda()
            img2 = data['target_img_data'].cuda()
            mods = data['mod']['str']

            optimizer.zero_grad()
            with autocast():
                loss = model.compute_loss(img1, mods, img2, cluster_result, front)
                total_loss = loss['rank'] \
                        + args.kappa_ * loss['kl_p'] \
                        + args.pro_w * loss['cls'] + args.kl_w * loss['kl_c']

            front = front + img1.shape[0]
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if i % args.save_summary_steps == 0:
                summary_batch = {}
                summary_batch['total_loss'] = total_loss.item()
                summ.append(summary_batch)
            loss_avg.update(total_loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()


def train_and_evaluate(model, optimizer, trainset, testset):
    trainloader = dataloader.DataLoader(trainset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                drop_last=True,
                                num_workers=args.num_workers)
    travelloader = dataloader.DataLoader(trainset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                drop_last=True,
                                num_workers=args.num_workers)
    
    current_best_score = float('-inf')
    best_parameters_model = None
    scaler = GradScaler()
    epoches = args.num_epochs
    tolerance = 0
    for epoch in range(epoches):
        cluster_result = None
        if epoch > 0 and (args.pro_w > 0 or args.kl_w > 0):
            clus_features = compute_features(travelloader, model, args)
            features = clus_features.numpy()
            cluster_result = run_kmeans_cpu(features, args)
        tolerance += 1
        if tolerance == 10:
            break
        if epoch !=0 and (epoch+1) % args.lr_decay == 0 and epoch < args.max_decay_epoch:
            for g in optimizer.param_groups:
                if g['name'] == 'clip':
                    g['lr'] *= args.clip_lr_div
                else:
                    g['lr'] *= args.lr_div

        logging.info("Epoch {}/{}".format(epoch + 1, epoches))
        train(model, optimizer, trainloader, scaler, epoch, cluster_result)
        current_score = 0
        current_result = []
        if args.dataset == 'fashioniq':
            for ci, category in enumerate(['dress', 'shirt', 'toptee']):
                t = test.test(args, model, trainset, category)
                logging.info(t)
                current_score = current_score + t[1][1]
                current_result.append(t)
            torch.save(model, os.path.join(args.model_dir, f'model_epoch_{epoch}.pt'))
            if current_score > current_best_score:
                current_best_score = current_score
                tolerance = 0
                best_json_path_combine = os.path.join(
                        args.model_dir, "metrics_best.json")
                test_metrics = {}
                
                for _ in current_result:
                    for metric_name, metric_value in _:
                        test_metrics[metric_name] = metric_value
                
                utils.save_dict_to_json(test_metrics, best_json_path_combine)
                best_parameters_model = model
        else:
            torch.save(model, os.path.join(args.model_dir, f'model_epoch_{epoch}.pt'))
            if args.dataset == 'shoes':
                t = test.test(args, model, trainset, 'shoes')
                logging.info(t)
                current_score = current_score + t[1][1] + t[2][1]
            elif args.dataset == 'cirr':
                t = test.test_cirr_valset(args, model, trainset)
                logging.info(t)
                current_score = t[0][1] + t[1][1] + t[2][1] + t[3][1] + t[4][1] + t[5][1] + t[6][1] 
            if current_score > current_best_score:
                current_best_score = current_score
                tolerance = 0
                best_json_path_combine = os.path.join(
                        args.model_dir, "metrics_best.json")
                test_metrics = {}
                for metric_name, metric_value in t:
                    test_metrics[metric_name] = metric_value
                utils.save_dict_to_json(test_metrics, best_json_path_combine)
                best_parameters_model = model 
        
    return current_best_score, test_metrics, best_parameters_model



if __name__ == '__main__':
    
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    import setproctitle

    proc_title = "python-c"
    setproctitle.setproctitle(proc_title) 
    print('Arguments:')
    for k in args.__dict__.keys():
        info = '    '+k+':'+str(args.__dict__[k])
        logging.info(info)

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info('Loading the datasets and model...')
    if args.dataset == "birds" or args.dataset == "fashion200k":
        trainset, testset = load_dataset()
    else:
        trainset = load_dataset()
        testset = None

    best_score = float('-inf')
    model, optimizer = create_model_and_optimizer()
    logging.info("Starting train for {} epoch(s)".format(args.num_epochs))
    _best_score, _metrics, current_model = train_and_evaluate(model, optimizer, trainset, testset)
    if _best_score > best_score:
        best_score = _best_score
        utils.save_dict_to_json(_metrics, os.path.join(args.model_dir, "metrics_best.json"))
        torch.save(current_model, os.path.join(args.model_dir, 'best_model.pt'))