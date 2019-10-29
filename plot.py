import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import json


if __name__ == "__main__":
    import sys
    log_dir = sys.argv[1]
    log_fnames = glob.glob("%s/*.log" % log_dir)
    average_ac_per_error = {}
    average_tac_per_error = {}
    std_ac_per_error={}
    std_tac_per_error={}
    for f_name in log_fnames:
        index = int(os.path.basename(f_name).split(".")[0])
        transition_state = np.unravel_index(index, [2 for i in     range(0, 8)])
        transition_matrix = np.zeros((2,2,2,2));
        for k, cur_value in enumerate(transition_state):
            cur_index = np.unravel_index(k, (2, 2, 2))
            transition_matrix[cur_index][0] = cur_value
            transition_matrix[cur_index][1] = 1 - cur_value
        with open (f_name, "r") as fp:
            for line in fp:
                line = line.strip()
                line_toks = line.split()
                if "Average for emission error" in line:
                    err = float(line_toks[4][:-1])
                    ac_acc = float(line_toks[7][:-1])
                    tac_acc = float(line_toks[10])
                    average_ac_per_error.setdefault(err, []).append(ac_acc)
                    average_tac_per_error.setdefault(err, []).append(tac_acc)
                if "STD for emission error" in line:
                    err = float(line_toks[4][:-1])
                    ac_acc = float(line_toks[7][:-1])
                    tac_acc = float(line_toks[10])
                    std_ac_per_error.setdefault(err,[]).append(ac_acc)
                    std_tac_per_error.setdefault(err,[]).append(tac_acc)
    with open("keras_log_summary.json", "w") as fp:
        json.dump({"a_ac": average_ac_per_error, "a_tac":average_tac_per_error, "s_ac": std_ac_per_error, "s_tac":std_tac_per_error}, fp)
        """
    for err in average_tac_per_error:
        x = average_ac_per_error[err]
        y = average_tac_per_error[err]
        x_err = std_ac_per_error[err]
        y_err = std_tac_per_error[err]
        plt.errorbar(x,y,xerr=x_err, yerr=y_err, fmt="ko")
        plt.savefig("%s.png" % err)

        """
