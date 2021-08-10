python train.py --gpu $1 --way 5 --shot 1 --train_way 5 --modules ICN_Loss --losses suppicnn --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 22
python train.py --gpu $1 --way 5 --shot 1 --train_way 30 --modules ICN_Loss --losses suppicnn --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 22
python train.py --gpu $1 --way 5 --shot 5 --train_way 5 --modules ICN_Loss --losses suppicnn --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 22
python train.py --gpu $1 --way 5 --shot 5 --train_way 20 --modules ICN_Loss --losses suppicnn --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 22

python train.py --gpu $1 --way 5 --shot 1 --train_way 5 --modules ICN_Loss --losses suppicnn,queryicnn --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 23
python train.py --gpu $1 --way 5 --shot 1 --train_way 30 --modules ICN_Loss --losses suppicnn,queryicnn --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 23
python train.py --gpu $1 --way 5 --shot 5 --train_way 5 --modules ICN_Loss --losses suppicnn,queryicnn --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 23
python train.py --gpu $1 --way 5 --shot 5 --train_way 20 --modules ICN_Loss --losses suppicnn,queryicnn --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 23