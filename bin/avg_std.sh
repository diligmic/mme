#! /bin/bash
FILE=/dev/stdin

if [[ -n $1 ]]; then FILE=$1; fi

cat $FILE | awk '/Acc/ {acc+=$NF; n+=1; V[n]=$NF}END{if (n>0) acc_avg=acc/n; acc_stddev=0.0; for(i in V) {acc_stddev+=(acc_avg-V[i])*(acc_avg-V[i]);} if (n>0) acc_stddev/=(n-1); acc_stddev=sqrt(acc_stddev); print "Accuracy: " acc_avg " (stddev: "acc_stddev ") (runs: "n")"}
                 /AUC/ {auc+=$NF; m+=1; W[m]=$NF}END{if (m>0) auc_avg=auc/m; auc_stddev=0.0; for(i in W) {auc_stddev+=(auc_avg-W[i])*(auc_avg-W[i]);} if (m>0) auc_stddev/=(m-1); auc_stddev=sqrt(auc_stddev); print "AUC-PR: " auc_avg " (stddev: "auc_stddev ") (runs: "m")"}'