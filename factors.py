import numpy as np

# P(dune) given prior
# States: t->True, m->mild, s->severe, n-> not present
cpt_0 = np.array([[0.5, 0.25, 0.25]])  # For dune = 'n', 'm', 's'. Only one row since 'oler' has one state in this context

# P(fori|dune) guessed piror
cpt_1 = np.array([
    [0.99, 0.01],  # For dune='n'
    [0.1, 0.9],  # For dune='m'
    [0.9, 0.1]  # For dune='s'
])

# P(dega|dune) guessed piror
cpt_2 = np.array([
    [0.99, 0.01],  # For dune='n'
    [0.9, 0.1],  # For dune='m'
    [0.1, 0.9]  # For dune='s'
])

# P(sloe|dune) guessed piror
cpt_3 = np.array([
    [0.995, 0.005],  # For dune='n'
    [0.5, 0.5],  # For dune='m'
    [0.5, 0.5]  # For dune='s'
])

# P(trim) given prior
cpt_4 = np.array([[0.9, 0.1]])  # Single row for 'oler' state 't'

# P(sloe|trim) guessed piror
cpt_5 = np.array([
    [0.99, 0.01],  # For trim='f'
    [0.995, 0.005]  # For trim='t'
])

cpts = [cpt_0, cpt_1, cpt_2, cpt_3, cpt_4, cpt_5]