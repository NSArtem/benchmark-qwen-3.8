#!/bin/zsh
# Memory guard for real-model runs: stop the target pid (and its children) if
# system free memory drops below 6% or swap use grows >3 GiB over its start.
# Usage: tools/memory_watchdog.sh <pid> <logfile>
pid=$1; log=$2
base=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'); base=${base%.*}
while kill -0 $pid 2>/dev/null; do
  free_pct=$(memory_pressure -Q | awk -F': ' '/free percentage/{gsub("%","",$2);print $2}')
  used=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'); used=${used%.*}
  echo "$(date +%T) free=${free_pct}% swap_used=${used}M" >> $log
  if (( free_pct < 6 )) || (( used - base > 3072 )); then
    echo "$(date +%T) SAFETY STOP free=${free_pct}% swap_used=${used}M" >> $log
    pkill -TERM -P $pid; kill -TERM $pid; exit 1
  fi
  sleep 5
done
