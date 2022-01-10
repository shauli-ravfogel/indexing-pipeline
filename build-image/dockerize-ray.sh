#!/bin/bash
cp Dockerfile.ray Dockerfile
bash ./dist.sh
gcloud builds submit --tag us.gcr.io/ai2-israel/spikeraygpu:spikeraygpu-encoding --timeout="4h0m00s" --machine-type="n1-highcpu-32"
