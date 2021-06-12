python test.py --way 5 --shot 1 --model_path pretrained/ConvNet-30w1s.pth --modules ICN --icn_original_score --icn_reduction_type supervised --tag 4
python test.py --way 5 --shot 1 --model_path pretrained/ConvNet-30w1s.pth --modules ICN --icn_original_score --icn_models umap,pca,isomap,kernel_pca,truncated_svd,feature_agg,fast_ica,nmf --tag 5
python test.py --way 5 --shot 1 --model_path pretrained/ConvNet-30w1s.pth --modules ICN --icn_original_score --icn_multiple_components --icn_n_dims 4 --icn_models umap,pca,isomap,kernel_pca,truncated_svd,feature_agg,fast_ica,nmf --tag 6
python test.py --way 5 --shot 5 --model_path pretrained/ConvNet-5w5s20t.pth --modules ICN --tag 1
python test.py --way 5 --shot 5 --model_path pretrained/ConvNet-5w5s20t.pth --modules ICN --icn_original_score --tag 2
python test.py --way 5 --shot 5 --model_path pretrained/ConvNet-5w5s20t.pth --modules ICN --icn_original_score --icn_multiple_components --icn_n_dims 4 --icn_reduction_type supervised --tag 3
python test.py --way 5 --shot 5 --model_path pretrained/ConvNet-5w5s20t.pth --modules ICN --icn_original_score --icn_reduction_type supervised --tag 4
python test.py --way 5 --shot 5 --model_path pretrained/ConvNet-5w5s20t.pth --modules ICN --icn_original_score --icn_models umap,pca,isomap,kernel_pca,truncated_svd,feature_agg,fast_ica,nmf --tag 5
python test.py --way 5 --shot 5 --model_path pretrained/ConvNet-5w5s20t.pth --modules ICN --icn_original_score --icn_multiple_components --icn_n_dims 4 --icn_models umap,pca,isomap,kernel_pca,truncated_svd,feature_agg,fast_ica,nmf --tag 6