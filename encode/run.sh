export GOOGLE_APPLICATION_CREDENTIALS="ai2-israel-d3744270e886.json"
export TOKENIZERS_PARALLELISM=false
ray down -n gpucluster --yes clust.yaml
ray up --yes clust.yaml
ray submit -v clust.yaml encode/encode_text.py --max_tokens_per_batch=16384 --skip_done
ray down -n gpucluster --yes clust.yaml
