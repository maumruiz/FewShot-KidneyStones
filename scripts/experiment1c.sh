python test.py --way 5 --shot 5 --gpu $1 --init_weights pretrained/Img900-Mini80-All.pth --dataset CrossKidneys --testset Daudon --ks_set mixed --tag 5
python test.py --way 5 --shot 20 --gpu $1 --init_weights pretrained/Img900-Mini80-All.pth --dataset CrossKidneys --testset Daudon --ks_set mixed --tag 5

python test.py --way 5 --shot 5 --gpu $1 --init_weights pretrained/Img900-All_ExcCUB.pth --dataset CrossKidneys --testset Daudon --ks_set mixed --tag 6
python test.py --way 5 --shot 20 --gpu $1 --init_weights pretrained/Img900-All_ExcCUB.pth --dataset CrossKidneys --testset Daudon --ks_set mixed --tag 6