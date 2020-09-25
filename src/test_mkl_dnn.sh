#!/usr/bin/sh
BENCH=../benchmarks/test_performance_intel.py

echo Testing lhapdf;
python ${BENCH} --lhapdf;

echo Testing w/o MKL-DNN;
python ${BENCH} --no_mkl;

echo Testing with inter_op 0 intra_op 0;
python ${BENCH} --inter_op 0 --intra_op 0;

echo Testing with inter_op 1 intra_op 96 \(default\);
python ${BENCH};

echo Testing with inter_op 2 intra_op 96;
python ${BENCH} --inter_op 2 --intra_op 96;

echo Plotting results;
python ${BENCH} --plot
