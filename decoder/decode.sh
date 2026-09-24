#!/bin/sh
# decode.sh <input> <output> <max seconds> <max wav bytes> <max address space>
# Prints the probed duration, exits 3 if it is over the limit, otherwise
# decodes to 16 kHz mono s16 WAV. -t and -fs bound the output even when the
# container's duration is missing or wrong; RLIMIT_FSIZE (exit 153) backs up
# -fs and RLIMIT_AS bounds memory.
set -u
in=$1 out=$2 max=$3 fs=$4 as=$5
dur=$(prlimit --as="$as" -- ffprobe -v error -show_entries format=duration -of csv=p=0 "$in" 2>/dev/null)
echo "duration=$dur"
case "$dur" in
  ''|N/A) ;;
  *) if awk -v d="$dur" -v m="$max" 'BEGIN { exit !(d > m) }'; then exit 3; fi ;;
esac
exec prlimit --fsize="$fs" --as="$as" -- \
  ffmpeg -nostdin -loglevel error -y -i "$in" -t $((max + 1)) -fs $((fs - 5000000)) \
  -ac 1 -ar 16000 -c:a pcm_s16le "$out"
