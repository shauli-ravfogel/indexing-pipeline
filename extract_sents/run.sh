export GOOGLE_APPLICATION_CREDENTIALS="../ai2-israel-d3744270e886.json"
ray stop --force
pipenv run ray up --yes clust.yaml
ray start --head
ray submit ../clust.yaml collect_sents.py --dir wikipedia
pipenv run ray down -n hiclusty --yes clust.yaml
