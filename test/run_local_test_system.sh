#!/bin/bash

# use case to branch and run cases
case "$1" in
  # first branch: one subpackage at a time: valid choices separated by pipe
  ard|flowers)
    CASES_TO_RUN=$1  # select-your-own
    ;;
  # default (no arg) behavior: run all
  ""|all)
    CASES_TO_RUN="ard flowers"  # default case ("all") -> ard then flowers
    ;;
  # error case: print error message and return >0
  *)
    echo "$1 NOT FOUND IN VALID CASES!"  # error case
    return 999  #
    ;;
esac

# cycle over indexed cases
IDX_CASE=0
for CASE in $CASES_TO_RUN; do
  FLAGS="--cov-report=html"  # default flag: html coverage report
  if [[ $IDX_CASE -gt 0 ]] ; then
    FLAGS="$FLAGS --cov-append"  # append after the first case for multiple
  fi
  if [[ "$CASE" == "ard" ]] && ! python -c "import optiwindnet" 2>/dev/null ; then
    FLAGS="$FLAGS --cov-config=.coveragerc_no_optiwindnet"  # special case
  fi
  pytest --cov=$CASE ${FLAGS} test/$CASE/system  # run the case
  ((IDX_CASE++))
done
