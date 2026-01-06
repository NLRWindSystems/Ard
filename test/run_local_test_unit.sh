#!/bin/bash
if python -c "import optiwindnet" 2>/dev/null ; then
  pytest --cov=ard --cov-report=html test/ard/unit
else
  pytest --cov=ard --cov-report=html test/ard/unit --cov-config=.coveragerc_no_optiwindnet
fi

rm -rf test/ard/unit/layout/problem*_out

#
