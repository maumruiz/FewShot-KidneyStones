python train.py --gpu $1 --way 5 --shot 1 --train_way 5 --modules ICN_Loss --losses cross,suppicnn,queryicnn --query_protos --tag 11
python train.py --gpu $1 --way 5 --shot 1 --train_way 30 --modules ICN_Loss --losses cross,suppicnn,queryicnn --query_protos --tag 11
python train.py --gpu $1 --way 5 --shot 5 --train_way 5 --modules ICN_Loss --losses cross,suppicnn,queryicnn --query_protos --tag 11
python train.py --gpu $1 --way 5 --shot 5 --train_way 20 --modules ICN_Loss --losses cross,suppicnn,queryicnn --query_protos --tag 11

python train.py --gpu $1 --way 5 --shot 1 --train_way 5 --modules ICN_Loss --losses suppicnn,queryicnn --query_protos --tag 12
python train.py --gpu $1 --way 5 --shot 1 --train_way 30 --modules ICN_Loss --losses suppicnn,queryicnn --query_protos --tag 12
python train.py --gpu $1 --way 5 --shot 5 --train_way 5 --modules ICN_Loss --losses suppicnn,queryicnn --query_protos --tag 12
python train.py --gpu $1 --way 5 --shot 5 --train_way 20 --modules ICN_Loss --losses suppicnn,queryicnn --query_protos --tag 12