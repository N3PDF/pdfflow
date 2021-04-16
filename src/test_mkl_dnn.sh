#!/usr/bin/sh
BENCH=../benchmarks/test_performance_intel.py

echo Generating samples ...;
python ${BENCH} -m generate;

echo Testing w/o MKL-DNN ...;
python ${BENCH} -m run --no_mkl;

#echo Testing with inter_op 0 intra_op 0 ...;
#python ${BENCH} --mode run --inter_op 0 --intra_op 0 --no_lhapdf;

echo Testing with inter_op 1 intra_op 96 ...;
python ${BENCH} -m run --inter_op 1 --intra_op 96 --no_lhapdf;

echo Testing with inter_op 2 intra_op 48 ...;
python ${BENCH} -m run --inter_op 2 --intra_op 48 --no_lhapdf;

echo Testing with inter_op 3 intra_op 32 ...;
python ${BENCH} -m run --inter_op 3 --intra_op 32 --no_lhapdf;

echo Plotting results ...;
python ${BENCH} -m plot
