export GOOGLE_APPLICATION_CREDENTIALS="../ai2-israel-d3744270e886.json"
ray stop --force
pipenv run ray up --yes clust.yaml
ray start --head
ray submit ../clust.yaml encode_text.py --max_tokens_per_batch=512 --output_path none
pipenv run ray down -n hiclusty --yes clust.yaml
