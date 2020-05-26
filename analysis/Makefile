evaluate:
	ccobra evaluate.json -s results/full_evaluation.csv | tee results/full_evaluation.out

plot:
	python3 plot_fig1_boxplot.py results/group.csv results/indiv.csv
	python3 plot_fig2_mReasoner_params.py results/mReasoner-fit.out
	python3 plot_fig3_phm_params.py results/phm-fit.out
	python3 plot_fig4_congruency.py results/indiv.csv
