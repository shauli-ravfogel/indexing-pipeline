export GOOGLE_APPLICATION_CREDENTIALS="ai2-israel-d3744270e886.json"
export TOKENIZERS_PARALLELISM=false
ray down -n gpucluster --yes clust.yaml
ray up --yes clust.yaml
ray submit -v clust.yaml encode/encode_text.py --max_tokens_per_batch=16384 --num_docs 25 --sub_dir wikipedia-sample-cls --sent_rep first
ray down -n gpucluster --yes clust.yaml

ray up --yes clust.yaml
ray submit -v clust.yaml encode/encode_text.py --max_tokens_per_batch=16384 --num_docs 25 --sub_dir wikipedia-sample-mean --sent_rep mean
ray down -n gpucluster --yes clust.yaml
