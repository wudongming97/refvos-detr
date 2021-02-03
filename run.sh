CUDA_VISIBLE_DEVICES=0 python train_video.py --masks --name exp4
CUDA_VISIBLE_DEVICES=1 python train_video.py --masks --name exp5

CUDA_VISIBLE_DEVICES=1 python train_video.py --masks --name exp48f
CUDA_VISIBLE_DEVICES=2 python train_video.py --masks --name exp48g
CUDA_VISIBLE_DEVICES=3 python train_video.py --masks --name exp50a2 --lr_drop 200

CUDA_VISIBLE_DEVICES=0 python train_video.py --masks --name exp61b
CUDA_VISIBLE_DEVICES=1 python train_video.py --masks --name exp53a --lr 2e-4
CUDA_VISIBLE_DEVICES=2 python train_video.py --masks --name exp53b --lr 2e-4

CUDA_VISIBLE_DEVICES=2 python train_video.py --masks --name exp60b

CUDA_VISIBLE_DEVICES=0 python train_video.py --masks --name exp61a
CUDA_VISIBLE_DEVICES=1 python train_video.py --masks --name exp61b
CUDA_VISIBLE_DEVICES=2 python train_video.py --masks --name exp61c

