python train.py --way 5 --shot 5 --gpu $1 --init_weights pretrained/imagenet900.pth --dataset CrossKidneys --trainset Cross --valset Daudon --testset Elbeze --ks_set mixed --model_name Cross-valDaudon --tag 10
python train.py --way 5 --shot 5 --gpu $1 --init_weights pretrained/imagenet900.pth --dataset CrossKidneys --trainset MiniImagenet --valset Daudon --testset Elbeze --ks_set mixed --model_name Img900-Mini80-valDaudon --tag 11