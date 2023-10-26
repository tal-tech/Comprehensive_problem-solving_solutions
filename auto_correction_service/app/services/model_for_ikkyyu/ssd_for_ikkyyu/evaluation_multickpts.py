# encoding: utf-8
import os
if __name__ == '__main__':
    model_path = '/workspace/OCR/models/train_models_s3d/checkpoint'
    with open(model_path, 'r') as f:
        ckpts = f.readlines()
    cfg_ckpt_paths = []
    for ckpt in ckpts[2:]:
        cfg_ckpt_paths.append(ckpt.split(':')[1][2:-2])
        print(ckpt.split(':')[1][2:-2])

    for cfg_ckpt_path in cfg_ckpt_paths:
#         print('python ssd_pr.py %s'%(cfg_ckpt_path))
        os.system('python /workspace/OCR/Layout_SSD/evaluation.py test select %s'%(cfg_ckpt_path))
    
