python trainval.py --way 5 --shot 5 --gpu 7 --dataset Cross --cross_ds CropDisease --backbone AmdimNet --max_epoch 80 --temperature 128 --step_size 20 --lr 0.0002 --init_weights pretrained/Img900-Mini80.pth --model_name Img900-Mini80-CD --tag 10
python trainval.py --way 5 --shot 5 --gpu 7 --dataset Cross --cross_ds Eurosat --backbone AmdimNet --max_epoch 50 --temperature 128 --step_size 20 --lr 0.0002 --init_weights pretrained/Img900-Mini80-CD.pth --model_name Img900-Mini80-CD-ES --tag 11
python trainval.py --way 5 --shot 5 --gpu 7 --dataset Cross --cross_ds ISIC --backbone AmdimNet --max_epoch 50 --temperature 128 --step_size 20 --lr 0.0002 --init_weights pretrained/Img900-Mini80-CD-ES.pth --model_name Img900-Mini80-CD-ES-ISIC --tag 12