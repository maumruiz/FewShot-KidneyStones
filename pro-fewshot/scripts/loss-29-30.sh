python train.py --gpu $1 --way 5 --shot 1 --train_way 5 --modules ICN_Loss --losses cross,prototriplet --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 29
python train.py --gpu $1 --way 5 --shot 1 --train_way 20 --modules ICN_Loss --losses cross,prototriplet --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 29
python train.py --gpu $1 --way 5 --shot 5 --train_way 5 --modules ICN_Loss --losses cross,prototriplet --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 29
python train.py --gpu $1 --way 5 --shot 5 --train_way 20 --modules ICN_Loss --losses cross,prototriplet --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 29

python train.py --gpu $1 --way 5 --shot 1 --train_way 5 --modules ICN_Loss --losses prototriplet --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 30
python train.py --gpu $1 --way 5 --shot 1 --train_way 20 --modules ICN_Loss --losses prototriplet --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 30
python train.py --gpu $1 --way 5 --shot 5 --train_way 5 --modules ICN_Loss --losses prototriplet --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 30
python train.py --gpu $1 --way 5 --shot 5 --train_way 20 --modules ICN_Loss --losses prototriplet --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 30