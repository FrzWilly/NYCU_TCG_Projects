```zsh
for i in {1..100}; do
	./2584 --total=100000 --block=1000 --limit=1000 --play="load=848586_ws.bin save=848586_ws.bin alpha=0.0035" | tee -a train.log
	./2584 --total=1000 --play="load=848586_ws.bin alpha=0" --save="stat.txt"
	tar zcvf weights.$(date +%Y%m%d-%H%M%S).tar.gz 848586_ws.bin train.log stat.txt
done
```

