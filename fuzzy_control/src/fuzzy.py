''' fuzzy.py
    --------

Implements fuzzy control in one degree of freedom.
'''

### Libraries
# Standard libraries
import numpy as np

# Third-party libraries
import skfuzzy as fuzz

# Related libraries 
import pid
import visualize
import matplotlib.pyplot as plt


class Fuzzy:
    def __init__(self):
        self.f_ssets = [[  # error
            [-10, -10, -5],  # -ve medium
            [-10, -5, 0],  # -ve small
            [-5, 0, 5],  # zero
            [0, 5, 10],  # +ve small
            [5, 10, 10],  # +ve medium
        ],
            # delta_error
            [
                [-10, -10, -5],  # -ve medium
                [-10, -5, 0],  # -ve small
                [-5, 0, 5],  # zero
                [0, 5, 10],  # +ve small
                [5, 10, 10],  # +ve medium
            ],
            # u
            [
                [-10, -10, -5],  # -ve medium
                [-10, -5, 0],  # -ve small
                [-5, 0, 5],  # zero
                [0, 5, 10],  # +ve small
                [5, 10, 10],  # +ve medium
            ]
        ]

        self.io_ranges = [  # range of e
            [-10, 10],
            # range of d_e
            [-10, 10],
            # range of u
            [-10, 10]
        ]

        self.mf_types = ['trimf', 'trimf', 'trimf']
        self.pre_error_r = 0
        self.inputs = [np.arange(var[0], var[1] + 5, 5) for var in self.io_ranges]
        print(self.inputs)
        self.b_array = []
        for i in range(3):
            self.b_array.append([self.membership_f(self.mf_types[i], self.inputs[i], a) for a in self.f_ssets[i]])
        visualize.visualize_mf(self.b_array, self.inputs)
        print(self.b_array)

    def reset(self):
        return 0

    def run_fuzzy_controller(self, error_r):
        if error_r > 160:
            error_r = 160
        elif error_r < -160:
            error_r = -160

        error = error_r / 16
        if error > 10.0:   error = 10.0
        if error < -10.0:   error = -10.0

        delta_e = (error_r - self.pre_error_r) / 16
        if delta_e > 10.0:   delta_e = 10.0
        if delta_e < -10.0:   delta_e = -10.0

        print("delta_e", delta_e)
        muval_e = self.fuzzify(self.inputs[0], self.b_array[0], error)
        muval_de = self.fuzzify(self.inputs[1], self.b_array[1], delta_e)

        f_mat = self.fuzzy_matrix(muval_e, muval_de)
        temp = np.copy(self.b_array)
        output = self.rule_base(temp, f_mat)
        aggregated = np.fmax(output[0], np.fmax(output[1], np.fmax(output[2], np.fmax(output[3], output[4]))))
        out_final = fuzz.defuzz(self.inputs[2], aggregated, 'centroid')
        self.pre_error_r = error_r
        return out_final

    def membership_f(self, mf, x, abc=[0, 0, 0], a=1, b=2, c=3, d=4, abcd=[0, 0, 0, 0]):
        """
        Returns y values corresponding to type of type of Membership fn.
        arguments:
            mf - string containing type of Membership function
            x  - x axis values
            abc - list containing triangular edge point x-values
        """
        return {
            'trimf': fuzz.trimf(x, abc),  # trimf(x, abc)
            'dsigmf': fuzz.dsigmf(x, a, b, c, d),  # dsigmf(x, b1, c1, b2, c2)
            'gauss2mf': fuzz.gauss2mf(x, a, b, c, d),  # gauss2mf(x, mean1, sigma1, mean2, sigma2)
            'gaussmf': fuzz.gaussmf(x, a, b),  # gaussmf(x, mean, sigma)
            'gbellmf': fuzz.gbellmf(x, a, b, c),  # gbellmf(x, a, b, c)
            'piecemf': fuzz.piecemf(x, abc),  # piecemf(x, abc)
            'pimf': fuzz.pimf(x, a, b, c, d),  # pimf(x, a, b, c, d)
            'psigmf': fuzz.psigmf(x, a, b, c, d),  # psigmf(x, b1, c1, b2, c2)
            'sigmf': fuzz.sigmf(x, a, b),  # sigmf(x, b, c)
            'smf': fuzz.smf(x, a, b),  # smf(x, a, b)
            'trapmf': fuzz.trapmf(x, abcd),  # trapmf(x, abcd)
            'zmf': fuzz.zmf(x, a, b),  # zmf(x, a, b)
        }[mf]

    def fuzzify(self, Input, y, crisp_val):
        """
        Fuzzifies crisp value to obtain their membership values for corr. fuzzy subsets.
        arguments:
            Input - Range of crisp_val i.e, list of x values discrete values which crisp_val can take
            y     - 2d list containing y values of each fuzzy subsets of an i/o variable
            crisp_val - value to be fuzzified
        """
        f = [fuzz.interp_membership(Input, fuzzy_sset_y, crisp_val) for fuzzy_sset_y in y]
        return f

    def fuzzy_matrix(self, muval_e, muval_de):
        """
        Returns 2d array of rule strengths
        arguments:
            muval_e, muval_de - 1d list of membership values to their corresponding fuzzy subsets
        """
        return np.array([[min(a, b) for a in muval_e] for b in muval_de])

    # b= y-values of trimf corresponding to each input and output variables in range var.ranges[]
    def rule_base(self, b, f_mat):
        """
        Returns y values of output by clipping by an amount of output activations for output fuzzy subsets
        arguments:
        f_mat - rule_strength matrix
        b     - b[2] , y values of output fuzzy subsets

    E / DEL_E|         NM      ||       NS        ||        Z         ||       PS        ||       PM         
    ----------------------------------------------------------------------------------------------------------
        NM   | f_mat[0][0] NM  || f_mat[0][1] NM  ||  f_mat[0][2] NS  || f_mat[0][3] Z   || f_mat[0][4] PS   
        NS   | f_mat[1][0] NM  || f_mat[1][1] NM  ||  f_mat[1][2] NS  || f_mat[1][3] PS  || f_mat[1][4] PM  
        Z    | f_mat[2][0] NM  || f_mat[2][1] NS  ||  f_mat[2][2] Z   || f_mat[2][3] PS  || f_mat[2][4] PM         
        PS   | f_mat[3][0] NM  || f_mat[3][1] NS  ||  f_mat[3][2] PS  || f_mat[3][3] PM  || f_mat[3][4] PM   
        PM   | f_mat[4][0] NS  || f_mat[4][1] Z   ||  f_mat[4][2] PS  || f_mat[4][3] PM  || f_mat[4][4] PM

        """
        NM = max(f_mat[0][0], f_mat[0][1], f_mat[1][0], f_mat[1][1], f_mat[2][0], f_mat[3][0])
        b[2][0] = np.fmin(NM, b[2][0])
        NS = max(f_mat[0][2], f_mat[1][2], f_mat[2][1], f_mat[3][1], f_mat[4][0])
        b[2][1] = np.fmin(NS, b[2][1])
        Z = max(f_mat[0][3], f_mat[2][2], f_mat[4][1])
        b[2][2] = np.fmin(Z, b[2][2])
        PS = max(f_mat[0][4], f_mat[1][3], f_mat[2][3], f_mat[3][2], f_mat[4][2])
        b[2][3] = np.fmin(PS, b[2][3])
        PM = max(f_mat[1][4], f_mat[2][4], f_mat[3][4], f_mat[3][3], f_mat[4][3], f_mat[4][4])
        b[2][4] = np.fmin(PM, b[2][4])

        return b[2]
