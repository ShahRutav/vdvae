# CUDA_VISIBLE_DEVICES=1 mpiexec -n 1 python testing.py --hps imagenet64 --test_eval
import numpy as np
import imageio
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
#from data import set_up_data
from utils import get_cpu_stats_over_ranks
from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
import socket
import argparse
from torch.nn.parallel.distributed import DistributedDataParallel
from utils import (logger,
                   local_mpi_rank,
                   mpi_size,
                   maybe_download,
                   mpi_rank)
#from mpi4py import MPI
import torch.distributed as dist
from data import mkdir_p
from vae import VAE
from torchvision.datasets import ImageFolder
import torchvision
import json
#from train_helpers import set_up_hyperparams, load_vaes, load_opt, accumulate_stats, save_model, update_ema

def set_up_data(H):
    shift_loss = -127.5
    scale_loss = 1. / 127.5
    if H.dataset == 'imagenet32':
        trX, vaX, teX = imagenet32(H.data_root)
        H.image_size = 32
        H.image_channels = 3
        shift = -116.2373
        scale = 1. / 69.37404
    elif H.dataset == 'imagenet64':
        #trX, vaX, teX = imagenet64(H.data_root)
        H.image_size = 64
        H.image_channels = 3
        shift = -115.92961967
        scale = 1. / 69.37404
    elif H.dataset == 'ffhq_256':
        trX, vaX, teX = ffhq256(H.data_root)
        H.image_size = 256
        H.image_channels = 3
        shift = -112.8666757481
        scale = 1. / 69.84780273
    elif H.dataset == 'ffhq_1024':
        trX, vaX, teX = ffhq1024(H.data_root)
        H.image_size = 1024
        H.image_channels = 3
        shift = -0.4387
        scale = 1.0 / 0.2743
        shift_loss = -0.5
        scale_loss = 2.0
    elif H.dataset == 'cifar10':
        (trX, _), (vaX, _), (teX, _) = cifar10(H.data_root, one_hot=False)
        H.image_size = 32
        H.image_channels = 3
        shift = -120.63838
        scale = 1. / 64.16736
    else:
        raise ValueError('unknown dataset: ', H.dataset)

    do_low_bit = H.dataset in ['ffhq_256']

    #if H.test_eval:
    #    print('DOING TEST')
    #    eval_dataset = teX
    #else:
    #    eval_dataset = vaX

    shift = torch.tensor([shift]).cuda().view(1, 1, 1, 1)
    scale = torch.tensor([scale]).cuda().view(1, 1, 1, 1)
    shift_loss = torch.tensor([shift_loss]).cuda().view(1, 1, 1, 1)
    scale_loss = torch.tensor([scale_loss]).cuda().view(1, 1, 1, 1)

    #if H.dataset == 'ffhq_1024':
    #    train_data = ImageFolder(trX, transforms.ToTensor())
    #    valid_data = ImageFolder(eval_dataset, transforms.ToTensor())
    #	untranspose = True
    #else:
    #    train_data = TensorDataset(torch.as_tensor(trX))
    #    valid_data = TensorDataset(torch.as_tensor(eval_dataset))
    untranspose = True

    def preprocess_func(x):
        nonlocal shift
        nonlocal scale
        nonlocal shift_loss
        nonlocal scale_loss
        nonlocal do_low_bit
        nonlocal untranspose
        'takes in a data example and returns the preprocessed input'
        'as well as the input processed for the loss'
        if untranspose:
            x[0] = x[0].permute(0, 2, 3, 1)
        inp = x[0].cuda(non_blocking=True).float()
        out = inp.clone()
        inp.add_(shift).mul_(scale)
        if do_low_bit:
            # 5 bits of precision
            out.mul_(1. / 8.).floor_().mul_(8.)
        out.add_(shift_loss).mul_(scale_loss)
        return inp, out

    return H, None, None, preprocess_func

def distributed_maybe_download(path, local_rank, mpi_size):
    print(path)
    #exit()
    if not path.startswith('gs://'):
        return path
    filename = path[5:].replace('/', '-')
    with first_rank_first(local_rank, mpi_size):
        fp = maybe_download(path, filename)
    return fp

def setup_mpi(H):
    H.mpi_size = mpi_size()
    H.local_rank = local_mpi_rank()
    H.rank = mpi_rank()
    os.environ["RANK"] = str(H.rank)
    os.environ["WORLD_SIZE"] = str(H.mpi_size)
    os.environ["MASTER_PORT"] = str(H.port)
    # os.environ["NCCL_LL_THRESHOLD"] = "0"
    os.environ["MASTER_ADDR"] = MPI.COMM_WORLD.bcast(socket.gethostname(), root=0)
    torch.cuda.set_device(H.local_rank)
    dist.init_process_group(backend='nccl', init_method=f"env://")

def save_model_txt(model, path):
    fout = open(path, 'w')
    for k, v in model.state_dict().items():
        fout.write(str(k) + '\n')
        fout.write(str(v.tolist()) + '\n')
    fout.close()

def save_model_json(model, path):
    from collections import OrderedDict
    actual_dict = OrderedDict()
    for k, v in model.state_dict().items():
      actual_dict[k] = v.tolist()
    with open(path, 'w') as f:
      json.dump(actual_dict, f)

def load_model_json(model, path):
  from collections import OrderedDict
  data_dict = OrderedDict()
  with open(path, 'r') as f:
    data_dict = json.load(f)    
  own_state = model.state_dict()
  print("Loading model...")
  for k, v in data_dict.items():
    #print('Loading parameter:', k)
    if not k in own_state:
      print('Parameter', k, 'not found in own_state!!!')
    if type(v) == list or type(v) == int:
      v = torch.tensor(v)
    own_state[k].copy_(v)
  model.load_state_dict(own_state)
  print('Model loaded')

def restore_params(model, path, local_rank, mpi_size, map_ddp=True, map_cpu=False):
    load_model_json(model, path)
    #### ORIGINAL CODE
    #state_dict = torch.load(distributed_maybe_download(path, local_rank, mpi_size), map_location='cpu' if map_cpu else None)
    #state_dict = torch.load(path, map_location='cpu' if map_cpu else None)
    #if map_ddp:
    #    new_state_dict = {}
    #    l = len('module.')
    #    for k in state_dict:
    #        if k.startswith('module.'):
    #            new_state_dict[k[l:]] = state_dict[k]
    #        else:
    #            new_state_dict[k] = state_dict[k]
    #    state_dict = new_state_dict
    #model.load_state_dict(state_dict, strict=False)

    #### SAVING MODEL
    #save_model_txt(model, 'encoder_wts.txt')
    #save_model_json(model, 'encoder_wts.json')

def load_vaes(H):
    vae = None
    #vae = VAE(H)
    #if H.restore_path:
    #    #logprint(f'Restoring vae from {H.restore_path}')
    #    print('Restoring vae from :', H.restore_path)
    #    restore_params(vae, H.restore_path, map_cpu=True, local_rank=None, mpi_size=None)

    ema_vae = VAE(H)
    if H.restore_ema_path:
        #logprint(f'Restoring ema vae from {H.restore_ema_path}')
        restore_params(ema_vae, H.restore_ema_path, map_cpu=True, local_rank=None, mpi_size=None)
    elif(vae):
        ema_vae.load_state_dict(vae.state_dict())
    ema_vae.requires_grad_(False)

    #vae = vae.cuda(H.local_rank)
    ema_vae = ema_vae.cuda(H.local_rank)

    #vae = DistributedDataParallel(vae, device_ids=[H.local_rank], output_device=H.local_rank)

    #if len(list(vae.named_parameters())) != len(list(vae.parameters())):
    #    raise ValueError('Some params are not named. Please name all params.')
    #total_params = 0
    #for name, p in vae.named_parameters():
    #    total_params += np.prod(p.shape)
    #print("Totat Params : ", total_params)
    #logprint(total_params=total_params, readable=f'{total_params:,}')
    return vae, ema_vae

def setup_save_dirs(H):
    H.save_dir = os.path.join(H.save_dir, H.desc)
    mkdir_p(H.save_dir)
    H.logdir = os.path.join(H.save_dir, 'log')

def set_up_hyperparams(s=None):
    H = Hyperparams()
    parser = argparse.ArgumentParser()
    parser = add_vae_arguments(parser)
    parse_args_and_update_hparams(H, parser, s=s) # hard code params inside this.
    #setup_mpi(H) # No MPI
    #setup_save_dirs(H) # No need to make a directory for logs.
    #logprint = logger(H.logdir)
    #for i, k in enumerate(sorted(H)):
    #    logprint(type='hparam', key=k, value=H[k])
    np.random.seed(H.seed)
    torch.manual_seed(H.seed)
    torch.cuda.manual_seed(H.seed)
    #logprint('training model', H.desc, 'on', H.dataset)
    #return H, logprint
    return H

def eval_step(data_input, target, ema_vae):
    with torch.no_grad():
        latents = ema_vae.get_latent_features(data_input)
    #stats = get_cpu_stats_over_ranks(stats)
    #print(latents.shape)
    return latents 

def main() : 
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image_path = "/home/rutav/latentRL/vdvae/imagenet.png"
    from PIL import Image
    im = Image.open(image_path)
    im = trans(im).unsqueeze(dim=0)
    print("Image Size : ", im.shape)
    im_list = [im]
    H = set_up_hyperparams()
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    vae, ema_vae = load_vaes(H)
    #print("Vae : ", vae)
    data_input, target = preprocess_fn(im_list)
    latents = eval_step(data_input, target, ema_vae)
    print(latents.shape)
	#run_test_eval(H, ema_vae, data_valid_or_test, preprocess_fn, logprint)

if __name__ == '__main__' :
	main()
