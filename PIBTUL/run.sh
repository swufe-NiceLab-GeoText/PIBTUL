#!/bin/bash
# PIBTUL training entry - run on a single city.
#   Usage:  ./run.sh [foursquare|gowalla|brightkite] [extra args...]
#   Example: ./run.sh foursquare --views OTR --rho 0.5 --seed 2024
# Defaults mirror the paper settings; override via CLI args.
set -u
cd "$(dirname "$0")"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

CITY="${1:-foursquare}"; shift
PY=/usr/local/iCompute/bin/python   # change to your torch interpreter if needed

case "$CITY" in
  foursquare)
    PROC=../data/Foursquare_traj_new.pkl
    TRAIN=../data/foursquare_6.txt
    VEC=../data/foursquare_embedding_node2vec_2.dat
    ;;
  gowalla)
    PROC=../data/gowalla_traj_2000_new.pkl
    TRAIN=../data/gowalla_2000.txt
    VEC=../data/gowalla_2000_embedding_node2vec.dat
    ;;
  brightkite)
    PROC=../data/brightkite_traj_new.pkl
    TRAIN=../data/brightkite_2000_5poi.txt
    VEC=../data/brightkite_embedding_node2vec.dat
    ;;
  *)
    echo "Unknown city: $CITY (choose foursquare|gowalla|brightkite)"; exit 1;;
esac

exec "$PY" main.py \
  --processed_file "$PROC" \
  --train_file "$TRAIN" \
  --vec_file "$VEC" \
  --city "$CITY" \
  --processed_flag \
  --epochs 80 "$@"
