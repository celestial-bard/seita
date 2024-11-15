#!/usr/bin/env bash

# Dataset URIs
# s3://argoverse/datasets/av2/sensor/ 
# s3://argoverse/datasets/av2/lidar/
# s3://argoverse/datasets/av2/motion-forecasting/
# s3://argoverse/datasets/av2/tbv/

# motion-forecasting; datasets
DATASET_NAME="$1"  # sensor, lidar, motion-forecasting or tbv.
TARGET_DIR="$2"    # Target directory on your machine.

if [ -z "$DATASET_NAME" ] || [ -z "$TARGET_DIR" ]; then
  echo "Usage: $0 <DATASET_NAME> <TARGET_DIR>"
  echo "Example: $0 motion-forecasting $HOME/datasets"
  exit 1
fi

s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/$DATASET_NAME/*" $TARGET_DIR
