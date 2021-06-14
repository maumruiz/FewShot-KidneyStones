python test.py --way 5 --shot 5 --gpu $1 --model_path pretrained/resnet-5w5s.pth --backbone ResNet12 --modules ICN --tag 7
python test.py --way 5 --shot 5 --gpu $1 --model_path pretrained/resnet-5w5s.pth --backbone ResNet12 --modules ICN --icn_original_score --tag 8
python test.py --way 5 --shot 5 --gpu $1 --model_path pretrained/resnet-5w5s.pth --backbone ResNet12 --modules ICN --icn_original_score --icn_multiple_components --icn_n_dims 6 --tag 9
python test.py --way 5 --shot 5 --gpu $1 --model_path pretrained/resnet-5w5s.pth --backbone ResNet12 --modules ICN --icn_original_score --icn_reduction_type supervised --icn_reduction_set support --tag 10
python test.py --way 5 --shot 5 --gpu $1 --model_path pretrained/resnet-5w5s.pth --backbone ResNet12 --modules ICN --icn_original_score --icn_models pca,isomap,kernel_pca,truncated_svd,feature_agg --tag 11
python test.py --way 5 --shot 5 --gpu $1 --model_path pretrained/resnet-5w5s.pth --backbone ResNet12 --modules ICN --icn_original_score --icn_multiple_components --icn_n_dims 6 --icn_models pca,isomap,kernel_pca,truncated_svd,feature_agg --tag 12