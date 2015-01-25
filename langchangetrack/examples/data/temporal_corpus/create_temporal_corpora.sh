SAMPLESIZE=100000
ls /scratch2/vvkulkarni/new_semantic/ngrams_expanded/eng-fiction/19*[0,5].ngrams | parallel -j16 --progress shuf -n $SAMPLESIZE {} -o {/}
