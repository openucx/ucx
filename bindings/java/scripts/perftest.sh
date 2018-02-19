#!/bin/bash

TOP_DIR="$(cd "$(dirname "$0")"/../"" && pwd)"
jarfile=$(find $TOP_DIR/target -name jucx*.jar)
classpath=.:$jarfile

cd $TOP_DIR/target/test-classes && java -cp $classpath org.ucx.jucx.perftest.Perftest "$@" 
