python trainval.py --way 5 --shot 5 --gpu 2 --dataset Cross --cross_ds MiniImagenet,CUB,CropDisease,Eurosat,ISIC,KidneyStones --backbone AmdimNet --max_epoch 400 --temperature 128 --step_size 20 --lr 0.0002 --init_weights pretrained/imagenet900.pth --model_name Img900-All --tag 20
python trainval.py --way 5 --shot 5 --gpu 2 --dataset Cross --cross_ds CropDisease,Eurosat,ISIC,KidneyStones --backbone AmdimNet --max_epoch 100 --temperature 128 --step_size 20 --lr 0.0002 --init_weights pretrained/Img900-Mini80.pth --model_name Img900-Mini80-All --tag 17
python trainval.py --way 5 --shot 5 --gpu 2 --dataset Cross --cross_ds MiniImagenet,CropDisease,Eurosat,ISIC,KidneyStones --backbone AmdimNet --max_epoch 200 --temperature 128 --step_size 20 --lr 0.0002 --init_weights pretrained/imagenet900.pth --model_name Img900-All_ExcCUB --tag 19
python trainval.py --way 5 --shot 5 --gpu 2 --dataset Cross --cross_ds CropDisease,Eurosat,ISIC,KidneyStones --backbone AmdimNet --max_epoch 100 --temperature 128 --step_size 20 --lr 0.0002 --init_weights pretrained/Img900-Mini80-CUB.pth --model_name Img900-Mini80-CUB-All --tag 18