python test.py --way 5 --shot 5 --gpu $1 --init_weights pretrained/imagenet900.pth --dataset CrossKidneys --testset Daudon --ks_set mixed --tag 1
python test.py --way 5 --shot 20 --gpu $1 --init_weights pretrained/imagenet900.pth --dataset CrossKidneys --testset Daudon --ks_set mixed --tag 1

python test.py --way 5 --shot 5 --gpu $1 --init_weights pretrained/Img900-Mini80.pth --dataset CrossKidneys --testset Daudon --ks_set mixed --tag 2
python test.py --way 5 --shot 20 --gpu $1 --init_weights pretrained/Img900-Mini80.pth --dataset CrossKidneys --testset Daudon --ks_set mixed --tag 2