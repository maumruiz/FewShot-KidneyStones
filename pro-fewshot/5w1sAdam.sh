python train.py --way 5 --shot 5 --gpu 2 --max_epoch 100 --backbone AmdimNet --optimizer Adam --temperature 128 --step_size 10 --lr 0.0001 --init_weights pretrained/imagenet900.pth
python train.py --way 5 --shot 5 --gpu 2 --max_epoch 100 --backbone AmdimNet --optimizer Adam --temperature 128 --step_size 10 --lr 0.0002 --init_weights pretrained/imagenet900.pth
python train.py --way 5 --shot 5 --gpu 2 --max_epoch 100 --backbone AmdimNet --optimizer Adam --temperature 128 --step_size 20 --lr 0.0001 --init_weights pretrained/imagenet900.pth
python train.py --way 5 --shot 5 --gpu 2 --max_epoch 100 --backbone AmdimNet --optimizer Adam --temperature 128 --step_size 20 --lr 0.0002 --init_weights pretrained/imagenet900.pth