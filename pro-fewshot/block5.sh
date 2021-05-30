python trainval.py --way 5 --shot 5 --gpu 7 --dataset Cross --cross_ds CropDisease --backbone AmdimNet --max_epoch 80 --temperature 128 --step_size 10 --lr 0.0002 --init_weights pretrained/imagenet1k.pth --model_name Img1k-CD --tag 25
python trainval.py --way 5 --shot 5 --gpu 7 --dataset Cross --cross_ds Eurosat --backbone AmdimNet --max_epoch 50 --temperature 128 --step_size 10 --lr 0.0002 --init_weights pretrained/Img1k-CD.pth --model_name Img1k-CD-ES --tag 26
python trainval.py --way 5 --shot 5 --gpu 7 --dataset Cross --cross_ds ISIC --backbone AmdimNet --max_epoch 50 --temperature 128 --step_size 10 --lr 0.0002 --init_weights pretrained/Img1k-CD-ES.pth --model_name Img1k-CD-ES-ISIC --tag 27