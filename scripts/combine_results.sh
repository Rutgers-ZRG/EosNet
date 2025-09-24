#!/bin/bash

# Script to combine original id_prop.csv with target_predicted.csv from prediction
# Creates a 3-column format like test_results.csv: struct_id, true_target, predicted_value

# Check if the user provided the required arguments
if [ $# -ne 3 ]; then
    echo "Usage: $0 <id_prop.csv> <target_predicted.csv> <output.csv>"
    echo "Combines original targets with predictions by matching struct_id"
    echo "Output format: struct_id, true_target, predicted_value"
    echo ""
    echo "Example: $0 id_prop_remain.csv target_predicted.csv combined_results.csv"
    exit 1
fi

ID_PROP_FILE=$1
PREDICTED_FILE=$2
OUTPUT_FILE=$3

# Check if input files exist
if [ ! -f "$ID_PROP_FILE" ]; then
    echo "Error: File '$ID_PROP_FILE' not found!"
    exit 1
fi

if [ ! -f "$PREDICTED_FILE" ]; then
    echo "Error: File '$PREDICTED_FILE' not found!"
    exit 1
fi

# Create temporary files for processing
TEMP_ORIGINAL=$(mktemp)
TEMP_PREDICTED=$(mktemp)
TEMP_JOINED=$(mktemp)

# Sort both files by struct_id (first column) for joining
sort -t',' -k1,1 "$ID_PROP_FILE" > "$TEMP_ORIGINAL"
sort -t',' -k1,1 "$PREDICTED_FILE" > "$TEMP_PREDICTED"

# Join the files on the first column (struct_id)
# Output format: struct_id, true_target (from original), predicted_value (from prediction)
join -t',' -1 1 -2 1 -o 1.1,1.2,2.2 "$TEMP_ORIGINAL" "$TEMP_PREDICTED" > "$TEMP_JOINED"

# Check if join was successful
if [ ! -s "$TEMP_JOINED" ]; then
    echo "Error: No matching struct_ids found between the two files!"
    echo "Please check that both files contain the same structure identifiers."
    rm -f "$TEMP_ORIGINAL" "$TEMP_PREDICTED" "$TEMP_JOINED"
    exit 1
fi

# Copy the joined result to output file
cp "$TEMP_JOINED" "$OUTPUT_FILE"

# Clean up temporary files
rm -f "$TEMP_ORIGINAL" "$TEMP_PREDICTED" "$TEMP_JOINED"

# Get statistics
ORIGINAL_COUNT=$(wc -l < "$ID_PROP_FILE")
PREDICTED_COUNT=$(wc -l < "$PREDICTED_FILE")
MATCHED_COUNT=$(wc -l < "$OUTPUT_FILE")

# Display results
echo "Successfully combined files:"
echo "- Original file: $ID_PROP_FILE ($ORIGINAL_COUNT entries)"
echo "- Predicted file: $PREDICTED_FILE ($PREDICTED_COUNT entries)"
echo "- Output file: $OUTPUT_FILE ($MATCHED_COUNT matched entries)"
echo ""

if [ "$MATCHED_COUNT" -lt "$ORIGINAL_COUNT" ]; then
    echo "Warning: Some entries from $ID_PROP_FILE were not found in $PREDICTED_FILE"
    echo "Missing entries: $((ORIGINAL_COUNT - MATCHED_COUNT))"
fi

if [ "$MATCHED_COUNT" -lt "$PREDICTED_COUNT" ]; then
    echo "Warning: Some entries from $PREDICTED_FILE were not found in $ID_PROP_FILE"
    echo "Extra predictions: $((PREDICTED_COUNT - MATCHED_COUNT))"
fi

echo ""
echo "Output format: struct_id, true_target, predicted_value"
echo "First 5 lines of $OUTPUT_FILE:"
head -n 5 "$OUTPUT_FILE"

if [ "$MATCHED_COUNT" -gt 5 ]; then
    echo "..."
    echo "Last 2 lines of $OUTPUT_FILE:"
    tail -n 2 "$OUTPUT_FILE"
fi
