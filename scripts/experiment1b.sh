python test.py --way 5 --shot 5 --gpu $1 --init_weights pretrained/Img900-Mini80-CD.pth --dataset CrossKidneys --testset Daudon --ks_set mixed --tag 3
python test.py --way 5 --shot 20 --gpu $1 --init_weights pretrained/Img900-Mini80-CD.pth --dataset CrossKidneys --testset Daudon --ks_set mixed --tag 3

python test.py --way 5 --shot 5 --gpu $1 --init_weights pretrained/Img900-Mini80-CUB-All.pth --dataset CrossKidneys --testset Daudon --ks_set mixed --tag 4
python test.py --way 5 --shot 20 --gpu $1 --init_weights pretrained/Img900-Mini80-CUB-All.pth --dataset CrossKidneys --testset Daudon --ks_set mixed --tag 4