import numpy as np
import matplotlib.pyplot as plt

def transition_rule(input, rule_bin):
    outputs = np.array([])
    for i in input:
        i = int(i)
        output = int(rule_bin[7 - i])
        outputs = np.append(outputs, output)
    return outputs


def cellular_automata(rule, row_length, N_iter, central_steps = None):
    rule_bin = bin(rule)[2:]
    if len(rule_bin) < 8:
        zeros_to_add = 8 - len(rule_bin)
        rule_bin = "0" * zeros_to_add + str(rule_bin)
    if central_steps != None:
        row = np.zeros(row_length)
        if central_steps%2 != 0:
            centre = int(row_length//2) 
            row[centre] = 1
            for i in range(central_steps //2):
                row[centre + i + 1] = 1
                row[centre - i - 1] = 1
        else: 
            centre = int(row_length//2) 
            row[centre] = 1
            row[centre-1] = 1
            for i in range((central_steps-2) //2):
                row[centre + i + 1] = 1
                row[centre - i - 2] = 1
    else:
        row = np.random.randint(2, size = row_length)

    results_array = np.zeros((N_iter + 1, row_length))
    results_array[0] = row

    for i in range(N_iter):
        neighbours = 4*np.roll(row, 1) + 2*np.roll(row,0) + 1*np.roll(row,-1)
        output = transition_rule(neighbours, rule_bin)
        results_array[i+1] = output
        row = output

    plt.imshow(results_array, cmap = "Greys")
    plt.title(f"Rule {rule}")
    plt.show()


#extra - reversible automata
def reversible_automata(rule, row1, row2, N_iter):
    rule_bin = bin(rule)[2:]
    if len(rule_bin) < 8:
        zeros_to_add = 8 - len(rule_bin)
        rule_bin = "0" * zeros_to_add + str(rule_bin)
    results_array = np.zeros((N_iter + 2, len(row1)))
    results_array[0] = row1
    results_array[1] = row2

    for i in range(N_iter):
        neighbours = 4*np.roll(row2, 1) + 2*np.roll(row2,0) + 1*np.roll(row2,-1)
        output = (transition_rule(neighbours, rule_bin) + row1)%2
        results_array[i+1] = output
        row1 = row2
        row2 = output

    plt.imshow(results_array, cmap = "Greys")
    plt.title(f"Rule {rule} reversed")
    plt.show()

if __name__ == '__main__':
    cellular_automata(rule = 30, row_length = 500, N_iter = 250, central_steps = 5)
    cellular_automata(rule = 22, row_length = 500, N_iter = 250, central_steps = 5)
    cellular_automata(rule = 30, row_length = 500, N_iter = 250, central_steps = None)
    cellular_automata(rule = 110, row_length = 500, N_iter = 250, central_steps = 5)

    #Extra - reversible automata:
    row_1 = [0,0,0,0,1,0,0,0,0]
    row_2 = [0,0,0,1,0,1,0,0,0]

    reversible_automata(rule = 90, row1 = row_1, row2 = row_2, N_iter = 100)

