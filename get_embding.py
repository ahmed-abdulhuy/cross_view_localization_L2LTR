import pickle
import os
import torch
import torch.nn as nn
import numpy as np


from utils.data_utils import get_loader
from torch.utils.data import DataLoader
from tqdm import tqdm
# from apex import amp
import scipy.io as scio
import torch.nn.functional as F
import argparse

from models.model_crossattn import VisionTransformer, CONFIGS
#from utils.data_utils import get_loader


#from utils.dataloader_act import TestDataloader
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def validate(dist_array, top_k):
    accuracy = 0.0
    data_amount = 0.0
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i,i]
        prediction = np.sum(dist_array[:, i] < gt_dist)
        if prediction < top_k:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount

    return accuracy



parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--name", required=True,
                    help="Name of this run. Used for monitoring.")
parser.add_argument("--dataset", choices=["CVUSA", "CVACT", "BBD"], default="CVUSA",
                        help="Which downstream task.")
parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16", "R50-ViT-B_32"],
                    default="R50-ViT-B_16",
                    help="Which variant to use.")
parser.add_argument("--polar", type=int,choices=[1,0],
                        default=0,
                        help="polar transform or not")
parser.add_argument("--dataset_dir", default="output", type=str,
                    help="The dataset path.")

parser.add_argument("--pretrained_dir", type=str, default="./models/EgoTR_model/CVUSA",
                        help="Where to search for pretrained ViT models.")

parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

parser.add_argument("--img_size", default=(128, 512), type=int,
                        help="Resolution size")

parser.add_argument("--img_size_sat", default=(128, 512), type=int,
                        help="Resolution size")

parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")

args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.n_gpu = torch.cuda.device_count()
args.device = device


config = CONFIGS[args.model_type]

model_grd = VisionTransformer(config, args.img_size)
model_sat = VisionTransformer(config, args.img_size_sat)




print("loading model form ", os.path.join(args.pretrained_dir,'model_grd_checkpoint.pth'))

state_dict = torch.load(os.path.join(args.pretrained_dir,'model_checkpoint.pth'))
model_grd.load_state_dict(state_dict['model_grd'])
model_sat.load_state_dict(state_dict['model_sat'])


from utils.dataloader_BBD import satDataloader, grdDataloader

satset = satDataloader(args)
grdset = grdDataloader(args)

sat_loader = DataLoader(satset,
                        batch_size=args.eval_batch_size,
                        shuffle=False, 
                        num_workers=4)

grd_loader = DataLoader(grdset,
                        batch_size=args.eval_batch_size,
                        shuffle=False, 
                        num_workers=4)


model_grd.to(device)
model_sat.to(device)

sat_global_descriptor = np.zeros([38026, 768])
grd_global_descriptor = np.zeros([190130, 768])
val_i =0

model_grd.eval()
model_sat.eval()


print('----------start creating sat embedding----------------------')
# start creat sat embedding
with torch.no_grad():
    for step, batch in enumerate(tqdm(sat_loader)):
        x_sat = batch
        if step == 1:
            print(x_sat.shape)

        x_sat=x_sat.to(args.device)

        sat_global = model_sat(x_sat)

        
        sat_global_descriptor[val_i: val_i + sat_global.shape[0], :] = sat_global.detach().cpu().numpy()

        val_i += sat_global.shape[0]


with open('./sat_discripto.pkl', 'wb') as f:
    pickle.dump(sat_global_descriptor, f)
print('--------------sat embedding is done-------------------')



# start creat ground embedding
val_i = 0
print('---------start creating ground embedding----------------')
with torch.no_grad():
    for step, batch in enumerate(tqdm(grd_loader)):
        x_grd = batch
        if step == 1:
            print(x_grd.shape)

        # print('===================shape', len(x_grd))

        x_grd=x_grd.to(args.device)

        grd_global = model_grd(x_grd)

        
        grd_global_descriptor[val_i: val_i + grd_global.shape[0], :] = grd_global.detach().cpu().numpy()

        val_i += grd_global.shape[0]



with open('./ground_discripto.pkl', 'wb') as f:
    pickle.dump(grd_global_descriptor, f)


print('='*50, '\n', '!Done')

# print('   compute accuracy')
# dist_array = 2.0 - 2.0 * np.matmul(sat_global_descriptor, grd_global_descriptor.T)
    
# top1_percent = int(dist_array.shape[0] * 0.01) + 1
# val_accuracy = np.zeros((1, top1_percent))

# print('start')

# for i in tqdm(range(top1_percent)):
#     val_accuracy[0, i] = validate(dist_array, i)


# print('top1', ':', val_accuracy[0, 1])
# print('top5', ':', val_accuracy[0, 5])
# print('top10', ':', val_accuracy[0, 10])
# print('top1%', ':', val_accuracy[0, -1])
