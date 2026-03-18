#!/bin/bash
# Download all model checkpoints for the 5 VBench dimensions:
#   aesthetic_quality, subject_consistency, overall_consistency,
#   motion_smoothness, dynamic_degree

set -e

CACHE_DIR="${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}"
echo "==> Model cache directory: $CACHE_DIR"

# ---------- 1. aesthetic_quality ----------
# CLIP ViT-L/14
mkdir -p "$CACHE_DIR/clip_model"
if [ ! -f "$CACHE_DIR/clip_model/ViT-L-14.pt" ]; then
    echo "[1/5] Downloading CLIP ViT-L/14 ..."
    wget -q --show-progress -P "$CACHE_DIR/clip_model" \
        "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
else
    echo "[1/5] CLIP ViT-L/14 already exists, skipping."
fi

# LAION aesthetic linear head
mkdir -p "$CACHE_DIR/aesthetic_model/emb_reader"
if [ ! -f "$CACHE_DIR/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth" ]; then
    echo "[1/5] Downloading aesthetic linear head ..."
    wget -q --show-progress -O "$CACHE_DIR/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth" \
        "https://hf-mirror.com/OpenGVLab/VBench_Used_Models/resolve/main/sa_0_4_vit_l_14_linear.pth"
else
    echo "[1/5] Aesthetic linear head already exists, skipping."
fi

# ---------- 2. subject_consistency ----------
# DINO repo (for local mode)
mkdir -p "$CACHE_DIR/dino_model"
if [ ! -d "$CACHE_DIR/dino_model/facebookresearch_dino_main" ]; then
    echo "[2/5] Cloning DINO repository ..."
    git clone --quiet https://github.com/facebookresearch/dino \
        "$CACHE_DIR/dino_model/facebookresearch_dino_main"
else
    echo "[2/5] DINO repository already exists, skipping."
fi

# DINO ViT-B/16 weights
if [ ! -f "$CACHE_DIR/dino_model/dino_vitbase16_pretrain.pth" ]; then
    echo "[2/5] Downloading DINO ViT-B/16 weights ..."
    wget -q --show-progress -P "$CACHE_DIR/dino_model" \
        "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
else
    echo "[2/5] DINO weights already exists, skipping."
fi

# ---------- 3. overall_consistency ----------
mkdir -p "$CACHE_DIR/ViCLIP"
if [ ! -f "$CACHE_DIR/ViCLIP/ViClip-InternVid-10M-FLT.pth" ]; then
    echo "[3/5] Downloading ViCLIP ..."
    wget -q --show-progress -P "$CACHE_DIR/ViCLIP" \
        "https://hf-mirror.com/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth"
else
    echo "[3/5] ViCLIP already exists, skipping."
fi

# ---------- 4. motion_smoothness ----------
mkdir -p "$CACHE_DIR/amt_model"
if [ ! -f "$CACHE_DIR/amt_model/amt-s.pth" ]; then
    echo "[4/5] Downloading AMT-S ..."
    wget -q --show-progress -P "$CACHE_DIR/amt_model" \
        "https://hf-mirror.com/lalala125/AMT/resolve/main/amt-s.pth"
else
    echo "[4/5] AMT-S already exists, skipping."
fi

# ---------- 5. dynamic_degree ----------
mkdir -p "$CACHE_DIR/raft_model"
if [ ! -f "$CACHE_DIR/raft_model/models/raft-things.pth" ]; then
    echo "[5/5] Downloading RAFT ..."
    wget -q --show-progress -P "$CACHE_DIR/raft_model" \
        "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"
    unzip -o -d "$CACHE_DIR/raft_model" "$CACHE_DIR/raft_model/models.zip"
    rm -f "$CACHE_DIR/raft_model/models.zip"
else
    echo "[5/5] RAFT already exists, skipping."
fi

echo ""
echo "==> All models downloaded to: $CACHE_DIR"
echo "    Total size:"
du -sh "$CACHE_DIR"
