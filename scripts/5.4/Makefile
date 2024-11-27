# Define the Python interpreter
PYTHON = python
T = 2
QS = 1e-5
E0S = 0.5
SPIN = 0.99
NAME = "horizon"

# Define the targets
all: run_spin run_eccentricity

kerr_eccentricity:
	@nohup $(PYTHON) produce_horizon_data.py -outname $(NAME) -Tobs $(T) -t kerr -wf kerr -qs $(QS) > logfiles/kerr_eccentricity.out &
	sleep 5

pn5_aak_eccentricity:
	@nohup $(PYTHON) produce_horizon_data.py -outname $(NAME) -Tobs $(T) -t pn5 -wf aak -qs $(QS) > logfiles/pn5_aak_eccentricity.out &
	sleep 5

kerr_aak_eccentricity:
	@nohup $(PYTHON) produce_horizon_data.py -outname $(NAME) -Tobs $(T) -t kerr -wf aak -qs $(QS) > logfiles/kerr_aak_eccentricity.out &
	sleep 5

kerr_spin: kerr_eccentricity
	@nohup $(PYTHON) produce_horizon_data.py -outname $(NAME) -Tobs $(T) -t kerr -wf kerr -qs $(QS) -e0s $(E0S) -grids spin M > logfiles/kerr_spin.out &
	sleep 5
	
pn5_aak_spin: pn5_aak_eccentricity
	@nohup $(PYTHON) produce_horizon_data.py -outname $(NAME) -Tobs $(T) -t pn5 -wf aak -qs $(QS) -e0s $(E0S) -grids spin M > logfiles/pn5_aak_spin.out &
	sleep 5
	
kerr_aak_spin: kerr_aak_eccentricity
	@nohup $(PYTHON) produce_horizon_data.py -outname $(NAME) -Tobs $(T) -t kerr -wf aak -qs $(QS) -e0s $(E0S) -grids spin M > logfiles/kerr_aak_spin.out &
	sleep 5
	
plot_eccentricity: kerr_eccentricity pn5_aak_eccentricity kerr_aak_eccentricity
	@$(PYTHON) plot_horizon_data.py -Tobs $(T) -q $(QS) -spin $(SPIN) -zaxis e0 -base $(NAME)

plot_spin: kerr_spin kerr_aak_spin 
	@$(PYTHON) plot_horizon_data.py -Tobs $(T) -q $(QS) -e0 $(E0S) -zaxis spin -base $(NAME)

run_eccentricity: kerr_eccentricity pn5_aak_eccentricity kerr_aak_eccentricity
	sleep 1

run_spin: kerr_spin pn5_aak_spin kerr_aak_spin
	sleep 1

	


# Clean target (optional)
clean:
	rm -f *.pyc

.PHONY: all kerr_eccentricity pn5_aak_eccentricity kerr_aak_eccentricity kerr_spin pn5_aak_spin kerr_aak_spin plot_eccentricity plot_spin clean