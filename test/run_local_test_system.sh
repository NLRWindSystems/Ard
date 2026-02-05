#!/bin/bash
if python -c "import optiwindnet" 2>/dev/null ; then
  pytest --cov=ard --cov-report=html test/ard/system
else
  pytest --cov=ard --cov-report=html test/ard/system --cov-config=.coveragerc_no_optiwindnet
fi

rm -rf test/ard/system/layout/problem*_out

#
