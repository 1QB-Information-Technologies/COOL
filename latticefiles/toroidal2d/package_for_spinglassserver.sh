
for L in 12; do
	rm -r sgs_tmp 2>/dev/null
	for dataset in validation; do

		LxL=${L}x${L}
		mkdir sgs_tmp

		for i in {00000000..00000099}; do
			awk "NR>"$((L*L+1))'{$1+=1; $2+=1; print $0}' $LxL/$dataset/$i/latfile > sgs_tmp/${LxL}xxx${dataset}xxx${i}.txt
		done
	done

	tar -cvf submit_${LxL}.tar sgs_tmp/*.txt
	rsync submit_${LxL}.tar fermi:~/Desktop/ 
	#rm submit_${LxL}.tar
	rm -r sgs_tmp

done





