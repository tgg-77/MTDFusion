# test phase
import torch
from torch.autograd import Variable
from net import *
import utils
from args_fusion import args
import numpy as np
import time
import cv2
import os
import scipy.io as io
import pytorch_msssim
from evaluation import *
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def load_model(path, input_nc, output_nc):
    print(path)
    model = model_generator(method='TDFusion')
    nest_model = model
    nest_model.load_state_dict(torch.load(path))

    para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

    nest_model.eval()
    nest_model.cuda()

    return nest_model


def _generate_fusion_image(model, img1, img2, img3):
    # encoder
    en_r = model.encoder(img1)
    en_v = model.encoder(img2)
    en_d = model.encoder(img3)
    # fusion
    f = model.fusion_1(en_r, en_v, en_d)
    # decoder
    img_fusion = model.decoder(f)
    return img_fusion


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    output_path_root = 'test_images/TNO/result/'
    if os.path.exists(output_path_root) is False:
        os.makedirs(output_path_root)
    model_path = './model/80000_sl1_ssim_96_1e3_gelu.model'
    in_c, out_c = 1, 1
    with torch.no_grad():
        model = load_model(model_path, in_c, out_c)
        out_path = 'test_images/TNO/mat'
        num = len(os.listdir(out_path))
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        en, ag, mssim, cen, qabf, mi, ssima = 0, 0, 0, 0, 0, 0, 0
        ori_time = time.time()
        time_list = []
        for i in range(num):
            index = i + 1
            img_path = os.path.join(out_path, '%d.mat' % (i + 1))
            cube = io.loadmat(img_path)['data']
            cube = np.float32(cube)
            cube = (cube - cube.min()) / (cube.max() - cube.min())
            cube = np.expand_dims(cube, axis=0)
            cube = torch.tensor(cube)
            if args.cuda:
                cube = cube.cuda()
            ir_img = cube[:, 0:1, :, :]
            vis_img = cube[:, 1:2, :, :]
            d_img = cube[:, 2:3, :, :]
            ir_img = Variable(ir_img, requires_grad=False)
            vis_img = Variable(vis_img, requires_grad=False)
            d_img = Variable(d_img, requires_grad=False)
            temp_time = time.time()
            img_fusion = _generate_fusion_image(model, ir_img, vis_img, d_img)
            temp_time_gap = time.time() - temp_time
            print(temp_time_gap)
            time_list.append(temp_time_gap)
            ############################ multi outputs ##############################################
            file_name = str(index) + '.png'
            output_path = output_path_root + file_name
            # # save images
            img_nol = img_fusion[0].cpu().data[0].numpy()
            img_nol = img_nol.transpose(1, 2, 0)
            img_nol = (img_nol-np.min(img_nol))/(np.max(img_nol)-np.min(img_nol))
            img_nol = img_nol * 255
            if args.cuda:
                img = img_fusion[0].cpu().clamp(0, 255).data[0].numpy()
            else:
                img = img_fusion[0].clamp(0, 255).data[0].numpy()
            img = img.transpose(1, 2, 0)
            utils.save_images(output_path, img_nol)
            utils.save_images(output_path, img)
            ori = cube.cpu().data[0].numpy()
            ori_ir = ori[0, :, :]
            ori_vis = ori[1, :, :]
            out = np.squeeze(img_nol).astype(np.uint8)
            ori_ir = (np.resize(ori_ir, (ori_ir.shape[0], ori_ir.shape[1]))*255).astype(np.uint8)
            ori_vis = (np.resize(ori_vis, (ori_vis.shape[0], ori_vis.shape[1]))*255).astype(np.uint8)
            i0 = cv2torch(out)
            i1 = cv2torch(ori_ir)
            i2 = cv2torch(ori_vis)
            ms = (float(ms_ssim(i0, i1)) + float(ms_ssim(i0, i2))) / 2
            en += Entropy(out)
            print(i, Entropy(out))
            ag += avgGradient(out)
            mssim += ms
            cen += (cross_entropy(out, ori_ir) + cross_entropy(out, ori_vis)) / 2
            qabf += Qabf(ori_ir, ori_vis, out)
            mi += MI(ori_ir, ori_vis, out)
            ssima += (float(ssim(i0, i1)) + float(ssim(i0, i2))) / 2
        time_list = time_list[1:]
        print('entropy:', en / num)
        print('AG:', ag / num)
        print('msssim:', mssim / num)
        print('cross_entropy:', cen / num)
        print('qabf:', qabf / num)
        print('MI:', mi / num)
        print('SSIMA:', ssima / num)
        print('time', np.mean(time_list), np.mean(time_list)-np.min(time_list), np.max(time_list)-np.mean(time_list))

    print('Done......')


if __name__ == '__main__':
    main()
