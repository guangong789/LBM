#!/bin/bash

SRC_BASE="$HOME/LBM/docs/nsight"
DST_BASE="/mnt/d/Docs/NsightDocs/LBM"

FOLDER="$1"

SRC="$SRC_BASE/$FOLDER"
DST="$DST_BASE/$FOLDER"

if [ ! -d "$SRC" ]; then
    echo "源目录不存在: $SRC"
    exit 1
fi

mkdir -p "$DST"

cp -rf "$SRC/"* "$DST/"

echo "$SRC -> $DST"