#!/bin/bash

# Set the directory containing the CSV files
dir=/mnt/d/SECRETed/3.3_MixMatch/Retrohosyntesys/lipidmaps/Tax-Table-LOTUSvsSECRETed

# Loop through each CSV file in the directory
for file in "$dir"/*.csv; do
  # Count the number of lines in the file
  num_lines=$(wc -l < "$file")

  # Check if the file has only one line
  if [ "$num_lines" -eq 1 ]; then
    # Delete the file if it has only one line
    rm "$file"
  fi
done
