"""
List of drugs supported by the simulator.
"""

DRUGS = [
    'PD0325901',
    'PLX-4720',
    'Selumetinib',
    'Lapatinib',
    'Erlotinib',
    'CHIR-265',
    'Vandetanib'
]

def empty_treatment():
    treatment = {
        'PD0325901': 0.0,
        'PLX-4720': 0.0,
        'Selumetinib': 0.0,
        'Lapatinib': 0.0,
        'Erlotinib': 0.0,
        'CHIR-265': 0.0,
        'Vandetanib': 0.0
    }
    return treatment

def single_treatment(drug, dosage):
    """
    Takes a drug name and dosage and returns a single drug treatment.
    """
    treatment = empty_treatment()
    assert drug in treatment, "Drug name is unknown."
    treatment[drug] = float(dosage)
    return treatment

def dual_treatment(ratio, dosage):
    """
    Takes a drug ratio and dosage and returns a dual drug treatment. Ratios are expected to be in percent.
    """
    treatment = empty_treatment()
    treatment['PD0325901'] = (1 - (ratio / 100.0)) * float(dosage)
    treatment['PLX-4720'] = (ratio / 100.0) * float(dosage)
    return treatment
