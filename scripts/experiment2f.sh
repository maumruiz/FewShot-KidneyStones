python test.py --way 4 --shot 5 --gpu $1 --init_weights pretrained/Cross-valDaudon.pth --dataset CrossKidneys --testset Estrade --ks_set mixed --tag 16
python test.py --way 4 --shot 5 --gpu $1 --init_weights pretrained/Img900-Mini80-valDaudon.pth --dataset CrossKidneys --testset Estrade --ks_set mixed --tag 17
python test.py --way 4 --shot 5 --gpu $1 --init_weights pretrained/Img900-Mini80-CD-valDaudon.pth --dataset CrossKidneys --testset Estrade --ks_set mixed --tag 18
python test.py --way 4 --shot 5 --gpu $1 --init_weights pretrained/Img900-Mini80-All-valDaudon.pth --dataset CrossKidneys --testset Estrade --ks_set mixed --tag 19
python test.py --way 4 --shot 5 --gpu $1 --init_weights pretrained/Img900-Mini80-CUB-valDaudon.pth --dataset CrossKidneys --testset Estrade --ks_set mixed --tag 20
python test.py --way 4 --shot 5 --gpu $1 --init_weights pretrained/Img900-Mini80-CUB-All-valDaudon.pth --dataset CrossKidneys --testset Estrade --ks_set mixed --tag 21