import argparse
import os
import sys
import shutil
import random
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from thop import profile
import logging
from utils import test_single_patch, find_Tissue_regions, create_folder, create_patches, \
    data_generator, fill_holes_wsi_seg
from network.DHUnet import DHUnet
from config import get_config
import pandas as pd
from PIL import Image
from datasets.dataset import DHUnet_dataset
from torchvision import transforms
Image.MAX_IMAGE_PIXELS = None

os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libffi.so.7'  # update this path
parser = argparse.ArgumentParser()

### network parameters
parser.add_argument('--volume_path', type=str,
                    default='train_dataset/test/', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='BCSS', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='train_dataset/test', help='list dir')
parser.add_argument('--pretrained_ckpt', type=str,
                    default='pretrained_ckpt/WSI-QA.pth', help='ckpt')
parser.add_argument('--output_dir', type=str, default='output', help='output dir')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--is_savenii', default=True, action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='output', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=301, help='random seed')
parser.add_argument('--cfg', type=str, default="configs/DHUnet_224.yaml", metavar="FILE", help='path to config file', )
parser.add_argument('--network', type=str, default='DHUnet', help='the model of network')
parser.add_argument('--fold_no', type=int, default=1, help='the i th fold')
parser.add_argument('--total_fold', type=int, default=5, help='total k fold cross-validation')
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--z_spacing', default=1, type=int, help='Test throughput only')

### wsi parameters (you may tune some of these parameters)
parser.add_argument('--batch_size', type=int, default=128, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=270, help='required tile size (Options:270, 540, or 1080)')
parser.add_argument('--wsilevel', default=0, type=int, help='level from open slide to read')
parser.add_argument('--thumbnail_size', default=5000, type=int, help='required wsi thumbnail resolution')
parser.add_argument('--wsi_folder', default="test_wsi", type=str, help='folder contains wsi images')
parser.add_argument('--cpu_workers', default=40, type=int, help='number of cpu workers')
parser.add_argument('--save_seg', default=1, type=int, help='to save tile segmentation result')
parser.add_argument('--back_thr', default=50, type=int, help='% of background to tolerate')
parser.add_argument('--blur_fold_thr', default=20, type=int, help='% of blur and fold to tolerate')


args = parser.parse_args()
config = get_config(args)
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])


def inference(args, model, test_loader, cpu_workers, test_save_path=None):
    all_tiles = []
    all_stats = []
    all_names = []
    for data, names in test_loader:
        data = data.cuda()
        batch_output_seg, batch_tile_sta = test_single_patch(args, data, model, cpu_workers, network=args.network)
        all_tiles.append(list(batch_output_seg))
        all_stats.append(batch_tile_sta)
        all_names.append(names)

    return all_tiles, all_stats, all_names


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    args.is_pretrain = True
    net = DHUnet(config, num_classes=args.num_classes)
    snapshot = args.pretrained_ckpt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net.to(device)
    msg = net.load_state_dict(torch.load(snapshot, map_location=device))
    print("self trained DHUnet ", msg)
    total = sum([param.nelement() for param in net.parameters()])
    input = torch.randn(1, 3, 224, 224).cuda()
    flops, params = profile(net, inputs=(input, input))[:2]
    snapshot_name = snapshot.split('/')[-1]
    log_folder = args.output_dir + '/test_log_/' + dataset_name
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    openslidelevel = args.wsilevel
    thumbnail_size = args.thumbnail_size
    tilesize = args.img_size
    data = args.wsi_folder
    cpu_workers = args.cpu_workers
    wsi_files = [f for f in os.listdir(data) if f.endswith(".svs") or f.endswith(".mrxs")]
    batch_size = args.batch_size
    tile_seg_sv = args.save_seg

    if tilesize != 270:
        assert "tile size larger than 270 is not recommended "


    start_time = time.time()

    for wsi_id, wsi_file in enumerate(wsi_files):

        stats = []
        stats.append(["tile", "%background", "%tissue", "%fold", "%blur", "classification"])
        ## generate output folders
        Qualified = os.path.join(data, wsi_file.split(".")[0]+"_results", "Qualified")
        Unqualified = os.path.join(data, wsi_file.split(".")[0]+"_results", "Unqualified")
        Tile_folder = os.path.join(data, wsi_file.split(".")[0]+"_results", "All_tiles")
        create_folder(Qualified)
        create_folder(Unqualified)
        create_folder(Tile_folder)

        ## generate tiles
        wsi_path = os.path.join(data, wsi_file)
        thumbnail, thumbnail_mask, thumbnail_roi, xmin_indx, ymin_indx, xmax_indx, ymax_indx, sf_w, sf_h = find_Tissue_regions(
            wsi_path, thumbnail_size, tilesize)
        create_patches(wsi_path, wsi_file, Tile_folder, cpu_workers, tilesize, xmin_indx, ymin_indx, xmax_indx,
                       ymax_indx)
        print("Tiles generation is done!")

        ## compute segmentation
        data_loader, total_patches = data_generator(Tile_folder, test_transform=test_transform,
                                                    batch_size=batch_size, worker=cpu_workers)
        output_seg, tile_stats, tile_names = inference(args, net, data_loader, cpu_workers, test_save_path)
        print("Tiles segmentation is done!")

        ## generate wsi segmentation mask
        thumbnail_h, thumbnail_w, _ = thumbnail.shape
        wsi_seg = np.zeros((thumbnail_h, thumbnail_w, 3), dtype=np.uint8)
        for btch_id, _ in enumerate(output_seg):
            batch_tile_name = tile_names[btch_id]
            batch_tile_img = output_seg[btch_id]
            tile_st = tile_stats[btch_id]

            for idx in range(0, len(batch_tile_name)):
                tile_img = batch_tile_img[idx]
                tile_name = batch_tile_name[idx]
                st = tile_st[idx]

                x_min_wsi = int(tile_name.split(".")[0].split("_")[-2])
                ymin_wsi = int(tile_name.split(".")[0].split("_")[-1])

                tile_img_resized = cv2.resize(tile_img, (int(tilesize / sf_w), int(tilesize / sf_h)),
                                              interpolation=cv2.INTER_NEAREST)
                if st[4] == "qualified":
                    source_path = os.path.join(Tile_folder, tile_name)
                    destination_path = os.path.join(Qualified, tile_name)
                    shutil.move(source_path, destination_path)
                    if tile_seg_sv:
                        tile_img_arr = Image.fromarray(tile_img)
                        tile_img_arr.save(os.path.join(Qualified, tile_name).split(".")[0] + "_seg.png")
                else:
                    source_path = os.path.join(Tile_folder, tile_name)
                    destination_path = os.path.join(Unqualified, tile_name)
                    shutil.move(source_path, destination_path)
                    if tile_seg_sv:
                        tile_img_arr = Image.fromarray(tile_img)
                        tile_img_arr.save(os.path.join(Unqualified, tile_name).split(".")[0] + "_seg.png")

                x_min_seg = int(x_min_wsi / sf_w)
                ymin_seg = int(ymin_wsi / sf_h)
                x_max_seg = int(x_min_seg + tile_img_resized.shape[0])
                y_max_seg = int(ymin_seg + tile_img_resized.shape[1])
                wsi_seg[ymin_seg:y_max_seg, x_min_seg:x_max_seg, :] = tile_img_resized
                stats.append([tile_name, st[0], st[1], st[2], st[3], st[4]])
        print("wsi segmentation mask is done!")

        ## save results
        wsi_seg_fill = fill_holes_wsi_seg(wsi_seg)
        thumbnail_mask_3d = np.repeat(thumbnail_mask[:, :, np.newaxis], 3, axis=2)
        masked_rgb_image = wsi_seg_fill * thumbnail_mask_3d
        masked_rgb_image = masked_rgb_image.astype(np.uint8)

        masked_rgb_image_arr = Image.fromarray(masked_rgb_image)
        masked_rgb_image_arr.save(
            os.path.join(data, wsi_file.split(".")[0] +"_results"+ "/" + wsi_file.split(".")[0] + "_seg.png"))

        thumbnail_arr = Image.fromarray(thumbnail)
        thumbnail_arr.save(os.path.join(data, wsi_file.split(".")[0] +"_results"+ "/" + wsi_file.split(".")[0] + "_thumbnail.png"))

        thumbnail_roi_arr = Image.fromarray(thumbnail_roi)
        thumbnail_roi_arr.save(
            os.path.join(data, wsi_file.split(".")[0]+"_results" + "/" + wsi_file.split(".")[0] + "_thumbnail_roi.png"))

        df = pd.DataFrame(stats[1:], columns=stats[0])
        excel_file_path = os.path.join(data, wsi_file.split(".")[0] +"_results"+ "/" + wsi_file.split(".")[0] + "_tile_stats.xlsx")
        df.to_excel(excel_file_path, index=False)
        shutil.rmtree(Tile_folder)
        print("results are saved")

        end_time = time.time()
        print((end_time - start_time)/60)


print("Completed !")
