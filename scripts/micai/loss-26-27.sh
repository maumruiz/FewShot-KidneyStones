python train.py --gpu $1 --way 5 --shot 1 --train_way 5 --modules ICN_Loss --losses cross,fullicnn --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 26
python train.py --gpu $1 --way 5 --shot 1 --train_way 20 --modules ICN_Loss --losses cross,fullicnn --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 26
python train.py --gpu $1 --way 5 --shot 5 --train_way 5 --modules ICN_Loss --losses cross,fullicnn --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 26
python train.py --gpu $1 --way 5 --shot 5 --train_way 20 --modules ICN_Loss --losses cross,fullicnn --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 26

python train.py --gpu $1 --way 5 --shot 1 --train_way 5 --modules ICN_Loss --losses fullicnn --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 27
python train.py --gpu $1 --way 5 --shot 1 --train_way 20 --modules ICN_Loss --losses fullicnn --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 27
python train.py --gpu $1 --way 5 --shot 5 --train_way 5 --modules ICN_Loss --losses fullicnn --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 27
python train.py --gpu $1 --way 5 --shot 5 --train_way 20 --modules ICN_Loss --losses fullicnn --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 27