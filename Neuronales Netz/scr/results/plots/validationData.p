#Gnuplot script to plot the validation data of several network architectures. One plot for each architecture, every plot contains the results for training with the flow equation method, neglecting the hessian and stochastic gradient descent

set xlabel 'Anzahl der Trainingsepochs'
set ylabel "Relative Genauigkeit bei der \n Klassifizierung der Validierungsdaten"
set style data lines
set term pdf
set key right bottom
set output 'validation_s_plot.pdf'
plot 'results_sgd_best_s.dat', 'results_fem_best_s.dat'
set key right center
set yrange[:100]
set output 'validation_vs_plot.pdf'
plot 'results_sgd_best_vs.dat', 'results_fem_best_vs.dat'
set output 'validation_vvs_plot.pdf'
plot 'results_sgd_best_vvs.dat', 'results_fem_best_vvs.dat'
set output 'validation_cps_plot.pdf'
plot 'results_sgd_best_cps.dat', 'results_fem_best_cps.dat'
set output 'validation_cpvs_plot.pdf'
plot 'results_sgd_best_cpvs.dat', 'results_fem_best_cpvs.dat'
set output 'validation_cpcpvs_plot.pdf'
plot 'results_sgd_best_cpcpvs.dat', 'results_fem_best_cpcpvs.dat'
set term x11
