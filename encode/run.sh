export GOOGLE_APPLICATION_CREDENTIALS="ai2-israel-d3744270e886.json"
ray down -n gpucluster --yes clust.yaml
ray up --yes clust.yaml
ray submit -v clust.yaml encode/encode_text.py --max_tokens_per_batch=512 --output_path none
ray down -n gpucluster --yes clust.yaml
