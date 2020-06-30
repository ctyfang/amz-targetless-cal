from prettytable import PrettyTable
import numpy as np

def compare_taus(tau_gt, tau_guess):
    tau_gt = np.squeeze(tau_gt)
    tau_guess = np.squeeze(tau_guess)

    x = PrettyTable()
    x.field_names = ["Id", "R1", "R2", "R3", "T1", "T2", "T3"]
    x.add_row(['GT']+tau_gt.tolist())
    x.add_row(['Guess']+tau_guess.tolist())
    print(x)