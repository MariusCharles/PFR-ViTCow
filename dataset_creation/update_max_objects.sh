#!/bin/bash

# =========================
# Usage check
# =========================
if [ $# -ne 2 ]; then
    echo "Usage: $0 <path_to_sam3_repo> <max_num_objects>"
    exit 1
fi

SAM3_PATH="$1"
MAX_OBJECTS="$2"
FILE="${SAM3_PATH}/sam3/model/sam3_video_base.py"

# =========================
# Check that the file exists
# =========================
if [ ! -f "$FILE" ]; then
    echo "Error: file $FILE not found."
    exit 1
fi

# =========================
# Check that the target line exists
# =========================
if ! grep -Eq '^[[:space:]]*max_num_objects[[:space:]]*=[[:space:]]*[0-9]+' "$FILE"; then
    echo "Error: target max_num_objects assignment not found in $FILE."
    exit 1
fi

# =========================
# Replace BOTH the config lines
# =========================
sed -i -E "s/^([[:space:]]*)max_num_objects[[:space:]]*=[[:space:]]*[0-9]+/\1max_num_objects = ${MAX_OBJECTS}/" "$FILE"
sed -i -E "s/^([[:space:]]*)num_obj_for_compile[[:space:]]*=[[:space:]]*[0-9]+/\1num_obj_for_compile = 4/" "$FILE"

# =========================
# Show result
# =========================
LINE_CONTENT1=$(grep -E '^[[:space:]]*max_num_objects[[:space:]]*=' "$FILE")
LINE_CONTENT2=$(grep -E '^[[:space:]]*num_obj_for_compile[[:space:]]*=' "$FILE")
echo "Updated line: $LINE_CONTENT1"
echo "Updated line: $LINE_CONTENT2"

echo "Modification completed successfully."
