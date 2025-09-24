#!/bin/bash

# Check if the user provided the required arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <filename> <N>"
    echo "Creates: <filename>_selected.<ext> (N random lines) and <filename>_remain.<ext> (remaining lines)"
    echo "Example: ./rnd_pick.sh id_prop.csv 20 creates id_prop_selected.csv and id_prop_remain.csv"
    exit 1
fi

FILENAME=$1
N=$2

# Check if the file exists
if [ ! -f "$FILENAME" ]; then
    echo "File not found!"
    exit 1
fi

# Check if N is a valid number
if ! [[ "$N" =~ ^[0-9]+$ ]]; then
    echo "Error: N must be a positive integer."
    exit 1
fi

# Get the total number of lines in the file
TOTAL_LINES=$(wc -l < "$FILENAME")

# Check if N is less than or equal to the total number of lines
if [ "$N" -gt "$TOTAL_LINES" ]; then
    echo "Error: N is greater than the total number of lines in the file."
    exit 1
fi

# Generate output filenames based on input filename
if [[ "$FILENAME" == *.* ]]; then
    # File has extension
    BASENAME="${FILENAME%.*}"
    EXTENSION="${FILENAME##*.}"
    SELECTED_FILE="${BASENAME}_selected.${EXTENSION}"
    REMAIN_FILE="${BASENAME}_remain.${EXTENSION}"
else
    # File has no extension
    SELECTED_FILE="${FILENAME}_selected"
    REMAIN_FILE="${FILENAME}_remain"
fi

# Use shuf to randomly select N lines and save to selected file
shuf -n "$N" "$FILENAME" > "$SELECTED_FILE"

# Create a temporary file for selected lines processing
TEMP_SELECTED=$(mktemp)
cp "$SELECTED_FILE" "$TEMP_SELECTED"

# Find remaining lines (not selected) and keep original order
# Use grep -F -v -f to exclude selected lines from original file
grep -F -v -f "$TEMP_SELECTED" "$FILENAME" > "$REMAIN_FILE"

# Clean up temporary files
rm -f "$TEMP_SELECTED"

# Display results
echo "Created files:"
echo "- $SELECTED_FILE: $N randomly selected lines"
echo "- $REMAIN_FILE: $(wc -l < "$REMAIN_FILE") remaining lines"
