python train.py --gpu $1 --way 5 --shot 1 --train_way 20 --modules ICN_Loss --losses cross,suppicnn,queryicnn --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 21
python train.py --gpu $1 --way 5 --shot 1 --train_way 20 --modules ICN_Loss --losses suppicnn --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 22
python train.py --gpu $1 --way 5 --shot 1 --train_way 20 --modules ICN_Loss --losses suppicnn,queryicnn --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 23
python train.py --gpu $1 --way 5 --shot 1 --train_way 20 --modules ICN_Loss --losses cross,suppicnn,queryicnn --query_protos --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 24
python train.py --gpu $1 --way 5 --shot 1 --train_way 20 --modules ICN_Loss --losses suppicnn,queryicnn --query_protos --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 25
python train.py --gpu $1 --way 5 --shot 1 --train_way 20 --modules ICN_Loss --losses cross,fullicnn --backbone ResNet12 --init_weights pretrained/ResNet-12-pretrained.pth --tag 26
