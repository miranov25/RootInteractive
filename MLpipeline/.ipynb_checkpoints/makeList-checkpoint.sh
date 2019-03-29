#!/usr/bin/env bash

input="/eos/user/r/rsadikin/data/JIRA/ATO-457/testconvergence/CPU/Distortion"
for a in $(ls $input/*.root); do
    echo Title:$a| sed -E  s_"(distortion|.root)"__g
    echo $a;
done > performance.list

