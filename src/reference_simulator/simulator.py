import os
import sys
import importlib
import amici
import pandas as pd
import numpy as np
from src.env.drugs import empty_treatment

MODEL_NAME = 'ERBB_RAS_AKT_Drugs'
CONDITIONS = pd.read_csv('./src/reference_simulator/conditions_petab.tsv', sep='\t')
ZERO_TREATMENT = empty_treatment()

sys.path.insert(0, os.path.abspath("") + "/src/reference_simulator/" + MODEL_NAME)
model_module = importlib.import_module(MODEL_NAME)

# NOTE: Below might be useful if we want to use simulator attributes later on
# amici.getSimulationStatesAsDataFrame(self.model, edatas, rdatas) 


class Simulator(object):
    """A wrapper class for Fabian's initial cancer cell model which provides some useful interfaces.

    A wrapper class which provides some useful interfaces for our purposes. The initialized class
    object has its own model and solver object which allows to run multiple instances in parallel.
    Before simulation the treatment plan one needs to initialize the simulator to load a specific
    cell line and reach an initial steady state. After that one applies the individual treatments.
    During the treatments on always runs until the next steady state has been reached. 

    Attributes:
        model: Private instance of cancer cell model.
        solver: Private instance of solver.
        initialized: Flag that marks if simulator has been initialized yet.
        R: Overall proliferation rate.
    """

    def __init__(self):
        ''' Instantiate simulator with private objects.'''
        self.model = model_module.getModel()
        self.solver = self.model.getSolver()
        self.zero_term = None
        self.line = None
        self.initialized = False
        self.R = -1
        

    def load_conditions(self, cell_line):
        """ Loads conditions for requested cell line into model

            Args:
                cell_line: String specifying cell line to simulate.

            Raises:
                ValueError: If specified cell line is unknown.
        """
        condition = CONDITIONS.loc[CONDITIONS.conditionId == f'TUMOR-{cell_line}-cellline-01-01', :]
        if len(condition) == 0:
            raise ValueError(f'Requested cell-line "{cell_line}" has no condition data.')
        for col in condition.columns:
            if col in self.model.getFixedParameterIds():
                self.model.setFixedParameterById(col, condition[col].values[0])

    def load_drug_concentrations(self, concentrations, verbose=False):
        """ Loads specified drug simulation into model.

            Args:
                concentrations: Dictionary specifying the concentrations of the 7 drugs.
                verbose: If set to true, prints drug concentrations.
        """
        for drug, conc in concentrations.items():
            parameter_id = self.model.getFixedParameterIds()[
                self.model.getFixedParameterNames().index(drug)
            ]
            self.model.setFixedParameterById(parameter_id, conc)
            if verbose:
                print(f'{drug}: {conc}')

    def initialize(self, cell_line):
        """Prepares experiment for requested cell line.

        Prepares simulator for experiment. It does though by giving the initial treatment zero
        treatment to the cell after which it runs the simulation until it reaches a steady state.
        Args:
            cell_line: String specifying cell line to simulate.

        Returns:
            relative_proliferation: The proliferation rate for the zero treatment which is always 1.

        Raises:
            ValueError: If provided cell line is unknown.
        """
        assert not self.initialized, "Simulator has already been initialized."
        if self.zero_term is not None:
            assert self.line == cell_line, "Need to modify initialization scheme in order to get new zero term."

        self.model.setTimepoints([np.infty])
        self.load_conditions(cell_line)
        self.load_drug_concentrations(ZERO_TREATMENT)

        # compute growth term for zero treatment only once
        if self.zero_term is None:
            edata_ref = amici.ExpData(self.model.get())
            edatas = [edata_ref]
            rdatas = amici.runAmiciSimulations(self.model, self.solver, edatas)
            self.zero_term = rdatas[0]["y"][0, 0]
            self.line = cell_line

        self.R = 1
        self.initialized = True
        return self.R 

    def apply_treatment(self, concentrations, verbose=False):
        '''Loads specified drug simulation into model and runs simulation for until next steady state.

        Assumes simulator has been initialzed for specific cell line first

        Args:
            concentrations: Specifies the drug cocktail.
            verbose: If set to true, prints drug concentrations.

        Returns:
            overall_proliferation_rate: Describes the overall proliferation of the cell line at the
                at the time at which the stable state has been reached.

        Raises:
            AssertionError: If simulator has not been initialized.
        '''
        assert self.initialized, "Simulator has not been initialized before first use."
        
        self.load_drug_concentrations(concentrations, verbose=verbose)
        
        edata_cond = amici.ExpData(self.model.get())
        edatas = [edata_cond]
        rdatas = amici.runAmiciSimulations(self.model, self.solver, edatas)

        cond_term = rdatas[0]["y"][0, 0]
        self.R  = self.R * (cond_term / self.zero_term)

        if verbose:
            print(f'time to steadystate {rdatas[0]["t_steadystate"]}')
            print(f'relative proliferation'f'{self.R}')

        return self.R 
