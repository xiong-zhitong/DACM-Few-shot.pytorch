#python train_dkl.py --backbone vgg16 --fold 0 --benchmark pascal --lr 1e-3 --bsz 1 --logpath "pascal-5i-fig" 
#python train.py --backbone vgg16 --fold 3 --benchmark pascal --lr 1e-3 --bsz 20 --logpath "pascal-5i-nnnf3-cos-sdt1" 
#python train_pgp.py --backbone resnet50 --fold 0 --benchmark pascal --lr 1e-3 --bsz 20 --logpath "resnet50-5i-sdt1" 
#python train_pgp.py --backbone vgg16 --fold 0 --benchmark pascal --lr 1e-3 --bsz 12 --logpath "pascal-5i-qmask-o" 
python train_pgp_bak.py --backbone vgg16 --fold 2 --benchmark pascal --lr 1e-3 --bsz 8 --logpath "pascal-5i-qmask-ocgpu-2" 
#python train.py --backbone vgg16 --fold 0 --benchmark pascal --lr 1e-3 --bsz 40 --logpath "pascal-5i-f1" &
#python train.py --backbone resnet50 --fold 0 --benchmark coco --lr 1e-5 --bsz 15 --logpath "coco-20i_base"
