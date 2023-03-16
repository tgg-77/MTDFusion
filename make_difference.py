import cv2
import os
import numpy as np
import scipy.io as io

path = 'test_images/TNO'

for i in range(30):
    d_name = os.path.join(path, 'D%d.png' % (i+1))
    ir_name = os.path.join(path, 'IR%d.png' % (i+1))
    vis_name = os.path.join(path, 'VIS%d.png' % (i+1))
    ir = cv2.imread(ir_name, 0)
    vis = cv2.imread(vis_name, 0)
    ir = ir.astype(np.float32)
    vis = vis.astype(np.float32)

    # method_1
    # d = cv2.subtract(ir, vis)

    # method_2
    # ir_new= (ir-ir.min())/(ir.max()-ir.min())
    # vis_new= (vis-vis.min())/(vis.max()-vis.min())
    # i_mean = np.mean(ir)
    # v_mean = np.mean(vis)
    # print(ir.shape, vis.shape)
    # d = (v_mean/i_mean)*ir - vis

    # method_3
    d = ir - vis
    d = abs(d)

    cv2.imwrite(d_name, d)
    cv2.imwrite(ir_name, ir)
    cv2.imwrite(vis_name, vis)