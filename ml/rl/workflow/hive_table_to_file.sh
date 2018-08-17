#!/usr/bin/env bash

# Usage:
#   flags:
#     -n for table namespace // required
#     -q for query string // required
#     -o for outfile path (must be .tsv file) // required
# Example:
#   bash ml/rl/workflow/hive_table_to_file.sh \
#     -n aml \
#     -q 'select * from rl_data_cartpole_v0_post_timeline where ds="2018-06-25"' \
#     -o ~/cartpole_data.tsv \

while getopts ":n:q:o:" opt; do
  case ${opt} in
    n ) TABLE_NAMESPACE=$OPTARG
      ;;
    q ) QUERY_STRING="set hive.cli.print.header=true; "$OPTARG
      ;;
    o ) OUT_FILE=$OPTARG
      ;;
    *)
      echo "Invalid option -$OPTARG" >&2
      exit
    ;;
  esac
done

printf "Executing:\\n\\n"
printf "hive --namespace $TABLE_NAMESPACE -e '$QUERY_STRING' > $OUT_FILE\\n\\n"
hive --namespace "$TABLE_NAMESPACE" -e "$QUERY_STRING" > "$OUT_FILE"
