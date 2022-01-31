export GOOGLE_APPLICATION_CREDENTIALS="ai2-israel-d3744270e886.json"
export TOKENIZERS_PARALLELISM=false
ray down -n gpucluster --yes clust.yaml
ray up --yes clust.yaml
ray submit -v clust.yaml encode/encode_text.py --max_tokens_per_batch=16384 --num_docs 50 --sub_dir wikipedia-sample-bert-base-cls --sent_rep cls
ray down -n gpucluster --yes clust.yaml

ray up --yes clust.yaml
ray submit -v clust.yaml encode/encode_text.py --max_tokens_per_batch=16384 --num_docs 50 --sub_dir wikipedia-sample-bert-base-mean --sent_rep mean
ray down -n gpucluster --yes clust.yaml

ray up --yes clust.yaml
ray submit -v clust.yaml encode/encode_text.py --max_tokens_per_batch=16384 --num_docs 50 --sub_dir wikipedia-sample-roberta-base-cls --sent_rep cls --model_name roberta-base
ray down -n gpucluster --yes clust.yaml

ray up --yes clust.yaml
ray submit -v clust.yaml encode/encode_text.py --max_tokens_per_batch=16384 --num_docs 50 --sub_dir wikipedia-sample-roberta-base-mean --sent_rep mean --model_name roberta-base
ray down -n gpucluster --yes clust.yaml

