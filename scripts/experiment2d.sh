python test.py --way 5 --shot 20 --gpu $1 --init_weights pretrained/Cross-valDaudon.pth --dataset CrossKidneys --testset Elbeze --ks_set mixed --tag 10
python test.py --way 5 --shot 20 --gpu $1 --init_weights pretrained/Img900-Mini80-valDaudon.pth --dataset CrossKidneys --testset Elbeze --ks_set mixed --tag 11
python test.py --way 5 --shot 20 --gpu $1 --init_weights pretrained/Img900-Mini80-CD-valDaudon.pth --dataset CrossKidneys --testset Elbeze --ks_set mixed --tag 12
python test.py --way 5 --shot 20 --gpu $1 --init_weights pretrained/Img900-Mini80-All-valDaudon.pth --dataset CrossKidneys --testset Elbeze --ks_set mixed --tag 13
python test.py --way 5 --shot 20 --gpu $1 --init_weights pretrained/Img900-Mini80-CUB-valDaudon.pth --dataset CrossKidneys --testset Elbeze --ks_set mixed --tag 14
python test.py --way 5 --shot 20 --gpu $1 --init_weights pretrained/Img900-Mini80-CUB-All-valDaudon.pth --dataset CrossKidneys --testset Elbeze --ks_set mixed --tag 15