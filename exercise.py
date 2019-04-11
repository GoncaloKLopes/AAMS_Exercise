import sys
import re
import copy
from scipy.optimize import linprog
import numpy as np
import math

OPTION_RE = "([a-zA-Z][0-9]*)=\(([0-9]+%?),(.+)\)"
LOTTERY_RE = "\[(([a-zA-Z][0-9]*=\([0-9]*%?,.+\),?)+)\]"
UPDATE_RE = "\((-?[0-9]+),(.+)\)"
ENCOUNTER_RE = "mine=\((.+)\),peer=\((.+)\)"
ENCOUNTER_UPDATE_RE = "mine=(-?[0-9]+),peer=(-?[0-9]+)"

DECIDE_RATIONAL = "decide-rational"
DECIDE_RISK = "decide-risk"
DECIDE_NASH = "decide-nash"
DECIDE_MIXED = "decide-mixed"
DECIDE_CONDITIONAL = "decide-conditional"
BLANK_STRING = "blank-decision"

PROB_TYPE = "p"
OCCUR_TYPE = "o"

SOLO_AGENT = 0
ENCOUNTER_AGENT = 1
###################### CLASS DEFINITIONS #######################################
class Agent:
    def __init__(self, type, state):
        if type in (DECIDE_RATIONAL, DECIDE_RISK):
            self.type = SOLO_AGENT
            self.state = parse_input(state)
        else:
            self.type = ENCOUNTER_AGENT
            self.state = string_to_bimatrix(state)

    def update(self, observation):
        if self.type == SOLO_AGENT:
            self.state = update_behavior(self.state, observation)

    def decide(self, behavior):
        if behavior == DECIDE_RATIONAL:
            return decide_rational(self.state)
        elif behavior == DECIDE_RISK:
            return decide_risk(self.state)
        elif behavior == DECIDE_NASH:
            return decide_nash(self.state)
        elif behavior == DECIDE_MIXED:
            return decide_mixed(self.state)
        else:
            return decide_conditional(self.state)


class Option:
    def __init__(self, id, prob, lottery):
        self.id = id
        self.prob = prob
        self.lottery = lottery
        self.type = PROB_TYPE if isinstance(self.prob, float) else OCCUR_TYPE

    def __repr__(self):
        prob = "".join([str(int(100 * self.prob)),"%"]) if isinstance(self.prob, float) else str(self.prob)
        return "".join([self.id, "=(", prob, ",", str(self.lottery), ")"])

    def has_nested_occurrences(self):
        if isinstance(self.lottery.options, int):
            return self.type == OCCUR_TYPE
        else:
            for option in self.lottery.options:
                if option.has_nested_occurrences():
                    return True

            return False

    def propagate_occur(self):
        if self.type == PROB_TYPE:
            raise Exception("Can't propagate if still a probability.")
        else:
            if self.lottery.type == None:
                return
            else:
                for option in self.lottery.options:
                    if option.type == PROB_TYPE:
                        option.type = self.type
                        option.prob = self.prob
                    option.propagate_occur()

class Lottery:
    def __init__(self, options):
        self.options = options
        if isinstance(self.options, int):
            self.type = None
        else:
            self.type = options[0].type

    def __repr__(self):
        res = "["
        if not self.type == None:
            return "".join(["[",",".join([str(option) for option in self.options]) ,"]"])
        else:
            return str(self.options)

    def __getitem__(self, idx):
        return self.options[idx]

    def get_utility(self):
        if isinstance(self.options, int):
            utility = self.options
        else:
            utility = 0
            if self.type == OCCUR_TYPE:
                occurrences = sum([option.prob for option in self.options])

            for option in self.options:
                prob = (option.prob / occurrences) if self.type == OCCUR_TYPE else option.prob
                utility += prob * option.lottery.get_utility()
        return utility

    def get_min_utility(self):
        if self.type == None:
            return self.options
        else:
            min = None
            for option in self.options:
                aux_min = option.lottery.get_min_utility()
                if min == None or aux_min < min:
                    min = aux_min
            return min

    def get_option(self, id):
        for option in self.options:
            if option.id == id:
                return option
        return None

    def add_option(self, option):
        options = options.append(option)


class Bimatrix:
    def __init__(self, ids, lott1, lott2):
        """
        lott1, lott2 -> numpy.ndarray of shape (2, 2) with lotteries for each choice
        """
        self.ids = ids
        self.len = len(self.ids)

        if type(lott1) != np.ndarray:
            raise Exception("lott1 is not an numpy.ndarray!")
        if type(lott2) != np.ndarray:
            raise Exception("lott2 is not an numpy.ndarray!")
        if lott1.shape != (self.len, self.len):
            raise Exception("lott1 has wrong shape", lott1.shape, "!")
        if lott2.shape != (self.len, self.len):
            raise Exception("lott2 has wrong shape", lott2.shape, "!")

        self.mat = np.empty((self.len, self.len, 2), dtype=Lottery)
        for i in range(self.len):
            for j in range(self.len):
                self.mat[i, j, 0] = lott1[i, j]
                self.mat[i, j, 1] = lott2[i, j]

        self.utilities_mat = np.zeros((self.len, self.len, 2))

        for i in range(self.len):
            for j in range(self.len):
                for k in range(2):
                    self.utilities_mat[i, j, k] = self.mat[i, j, k].get_utility()

    def __getitem__(self, idx):
        i, j = idx
        return self.utilities_mat[i, j]

    def __repr__(self):
        return str((self.ids, self.mat))

    def nash_equilibrium(self):
        """
        Returns a list of tuples with 2 elements: ((T0, T1) (3, 4))
        """
        res = []
        nash = np.zeros((self.len, self.len, 2))

        maxes = np.empty((2, self.len, self.len))

        for i in range(self.len):
                maxes[0, i] = max_indices([u[0] for u in self.utilities_mat[:, i]])
        for i in range(self.len):
                maxes[1, i] = max_indices([u[1] for u in self.utilities_mat[i, :]])

        #check nash for p1
        for i in range(self.len):
            for j in range(self.len):
                nash[i, j, 0] = maxes[0, i, j]

        #check nash for p2
        for i in range(self.len):
            for j in range(self.len):
                nash[i, j, 1] = maxes[1, i, j]

        for i in range(self.len):
            for j in range(self.len):
                if (nash[i, j, 0] and nash[i, j, 1]):
                    res.append(((self.ids[i], self.ids[j]), tuple(self.utilities_mat[i, j, k] for k in range(2))))
        return res
################################################################################

####################### INPUT PARSING FUNCTIONS ################################
def string_to_option(string):
    p = re.compile("([a-zA-Z][0-9]*)=\(([0-9]*%?),(.+)\)")
    m = p.match(string)
    if not m:
        raise Exception("Bad input string") #create an adequate exception
    id = m.group(1)

    prob = m.group(2)
    if "%" in prob:
        prob = 0.01 * int(m.group(2)[:-1])
    else:
        prob = int(m.group(2))

    lottery = string_to_lottery(m.group(3))

    return Option(id, prob, lottery)

def string_to_lottery(string):
    if re.match("-?[0-9]+", string):
        return Lottery(int(string))
    options = []
    curr_str = ""
    start = 1
    cont = 1
    while(cont):
        for i in range(start, len(string)-1):
            if string[i] != "(":
                curr_str = "".join([curr_str, string[i]]) #append chars until a "(" is found
            else:
                matching_idx = find_matching_parenthesis(string, i)
                curr_str = "".join([curr_str, string[i:matching_idx+1]])
                start = matching_idx + 2 #start parsing again after ,
                if start > (len(string) - 1): #check if string has ended
                    cont = 0
                options.append(curr_str)
                curr_str = ""
                break

    for i in range(len(options)):
        options[i] = string_to_option(options[i])

    return Lottery(options)


def string_to_bimatrix(string):
    p = p = re.compile(ENCOUNTER_RE)
    m = p.match(string)
    mine, peer = m.groups()

    plotts = [[], []]
    tids = []

    for pstring, plott in zip((mine, peer), plotts):
        pmode = 0
        cont = 1
        start = 0
        curr_lotts = []
        while cont:
            tid_aux0 = ""
            tid_aux1 = ""
            for i in range(start, len(pstring)):
                #print(pstring[i])

                if pstring[i] != "|" and pstring[i] != "=":
                    if not pmode:
                        tid_aux0 += pstring[i]

                    else:
                        tid_aux1 += pstring[i]

                elif pstring[i] == "|":
                    if not tid_aux0 in tids:
                        tids.append(tid_aux0)
                    pmode = 1

                elif pstring[i] == "=":
                    if not tid_aux1 in tids:
                        tids.append(tid_aux1)
                    pmode = 0

                    matching_idx = find_matching_parenthesis(pstring, i + 1)
                    curr_lotts.append(string_to_lottery(pstring[i+1:matching_idx+1]))
                    start = matching_idx + 2
                    if start >= len(pstring):
                        cont = 0
                    break

        plott.append(split_list_equals(curr_lotts))

    for i in range(len(plotts)):
        plotts[i] = np.squeeze(plotts[i])

    return Bimatrix(tuple(tids), np.array(plotts[0], dtype=Lottery), np.array(plotts[1], dtype=Lottery))

def parse_input(inpt):
    tasks = {}
    curr_id = ""
    cont = 1
    start = 1
    while cont:
        for i in range(start, len(inpt)): #remove parenthesis
            if inpt[i] != "=":
                curr_id = "".join([curr_id, inpt[i]])
            else:
                matching_idx = find_matching_parenthesis(inpt, i+1)
                tasks[curr_id] = string_to_lottery(inpt[i+1:matching_idx+1])
                start = matching_idx + 2
                if start > (len(inpt) - 1):
                    cont = 0
                curr_id = ""
                break
    return tasks
################################################################################
####################### DECISION FUNCTIONS######################################
def decide_rational(tasks):
    """
    tasks is a dict
    """
    max_util = None
    for key in tasks:
        aux_max = tasks[key].get_utility()
        if (max_util == None) or aux_max > max_util[1]: #doesn't check second if first is True
            max_util = (key, aux_max)
    return max_util[0]

def update_behavior(tasks, inpt):
    pat = re.compile(UPDATE_RE)
    match = pat.match(inpt)

    new_lottery = Lottery(int(match.group(1)))
    tiers = (match.group(2)).split(".")
    new_tasks = copy.deepcopy(tasks)
    curr_lott = new_tasks[tiers[0]]
    new_options = []

    if len(tiers) > 1:
        for tier in tiers[1:-1]:
            #as we progress through the tiers, check every option and verify if
            #it's a probability or if it has nested occurrence options,
            #creating a new list of updated options
            for option in curr_lott.options:
                if option.id == tier:
                    if option.type == OCCUR_TYPE:
                        option.prob += 1
                    else:
                        option.prob = 1
                        option.type = OCCUR_TYPE
                        curr_lott.type = OCCUR_TYPE
                        option.propagate_occur()
                elif option.type == OCCUR_TYPE:
                    pass #will be appended after this big if
                elif option.has_nested_occurrences() and option.type == PROB_TYPE:
                    option.prob = 0
                    option.type = OCCUR_TYPE
                    option.propagate_occur()
                else:
                    continue
                new_options.append(option)
            curr_lott.options = new_options
            new_options = []
            curr_lott = (curr_lott.get_option(tier)).lottery


    if curr_lott.type == PROB_TYPE or curr_lott.type == None:
        curr_lott.options = [Option(tiers[-1], 1, new_lottery)]
        curr_lott.type = OCCUR_TYPE
        curr_lott.prob = 1
    else:
        old_option = curr_lott.get_option(tiers[-1])
        if old_option == None:
            (curr_lott.options).append(Option(tiers[-1], 1, new_lottery))
        else:
            old_option.prob += 1

    return new_tasks

def decide_risk(tasks):
    res_str = ""
    str_format = "{.2f}"
    l = len(tasks)

    c = np.zeros(l)
    A_ub = np.zeros((1, l))
    A_eq = np.ones((1, l))
    b_ub = np.array([[0]])
    b_eq = np.array([[1]])

    for i, task_id in enumerate(tasks):
        c[i] = -1 * tasks[task_id].get_utility()
        A_ub[0, i] = -1 * tasks[task_id].get_min_utility()

    eq_pairs = get_equal_pairs(c)

    for pair in eq_pairs:
        if (A_ub[0, pair[0]] <= 0) != (A_ub[0, pair[1]] <= 0):
            continue
        aux = np.zeros((1, l))
        aux[0, pair[0]] = 1
        aux[0, pair[1]] = -1
        A_eq = np.vstack((A_eq, aux))
        b_eq = np.vstack((b_eq, 0))

    all_neg = True
    for el in A_ub[0]:
        if el < 0:
            all_neg = False
            break
    if not all_neg:
       # print(linprog(c, method="simplex", A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub = b_ub))
        contribs = linprog(c, method="simplex", A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub = b_ub).x
    else:
        #print(linprog(c, method="simplex", A_eq=A_eq, b_eq=b_eq))
        contribs = linprog(np.squeeze(A_ub, axis=0), method="simplex", A_eq=A_eq, b_eq=b_eq).x

    if isinstance(contribs, float):
        contribs = np.array(contribs)
    #print(contribs)
    #this is huge but the tuple comprehension inside the list comprehension prevents extra memory allocation
    res_str = ";".join([pair for
                        pair in (",".join([str(round(contrib, 2)),task]) for
                        contrib,task in zip(contribs, tasks.keys()) if contrib > 0)])
    return "(" + res_str + ")"

def decide_nash(bimatrix):
    nashes = bimatrix.nash_equilibrium()
    max_nash = 0
    mine = ""
    peer = ""
    #print(nashes)
    if len(nashes) > 1:
        for nash in nashes:
            aux_max = nash[1][0] + nash[1][1]
            if aux_max > max_nash:#only updates if higher, otherwise keeps ordered by idx
                mine = nash[0][0]
                peer = nash[0][1]
                max_nash = aux_max
    elif len(nashes) == 0:
        return BLANK_STRING
    else:
        mine = nashes[0][0][0]
        peer = nashes[0][0][1]

    return "mine=" + mine + ",peer=" + peer

def decide_mixed(bimatrix):
    alpha = 0
    beta = 0

    alpha = (bimatrix[1, 1][1] - bimatrix[1, 0][1]) /\
    (bimatrix[0, 0][1] - bimatrix[1, 0][1] - bimatrix[0, 1][1] + bimatrix[1, 1][1])
    beta = (bimatrix[1, 1][0] - bimatrix[0, 1][0]) /\
    (bimatrix[0, 0][0] - bimatrix[0, 1][0] - bimatrix[1, 0][0] + bimatrix[1, 1][0])

    alpha = round(alpha, 2)
    beta = round(beta, 2)
    if alpha > 1 or beta > 1:
        return BLANK_STRING
    else:
        return "mine=(" + str(alpha) + "," + str(1-alpha) + "),peer=(" + str(beta) + "," + str(1-beta) + ")"

def decide_conditional(bimatrix):
    res = decide_nash(bimatrix)
    if res == BLANK_STRING:
        return decide_mixed(bimatrix)
    else:
        return res
################################################################################

####################### AUXILIARY FUNCTIONS ####################################
def opposite_par(par):
    if par == "{":
        return "}"
    elif par == "(":
        return ")"
    elif par == "[":
        return "]"
    return False

def find_matching_parenthesis(string, idx):
    par = string[idx]
    if par not in ("{", "(", "["):
        raise Exception("Not a bracket.")
    lvl = 0
    for i in range(idx + 1, len(string)):
        if string[i] == opposite_par(par):
            if lvl == 0:
                return i
            else:
                lvl -= 1
        elif string[i] == par:
            lvl += 1
    return False

def get_equal_pairs(lst):
    pairs = []
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            if lst[i] == lst[j]:
                pairs.append([i, j])
    return pairs

def max_indices(lst):
    m = max(lst)
    return [(x == m) * 1 for x in lst]

def split_list_equals(lst):
    #print(lst)
    chunk = int(math.sqrt(len(lst)))
    if len(lst) % chunk != 0:
        raise Exception("Not splittable in equal parts.")

    return [lst[(i*chunk):((i+1)*chunk)] for i in range(chunk)]
################################################################################


args = sys.stdin.readline().split(' ')
agent = Agent(args[0], args[1])
size = 1 if len(args) <= 2 else int(args[2])
for i in range(0, size):
    if i != 0:
        agent.update(sys.stdin.readline())
    sys.stdout.write(agent.decide(args[0]) + '\n')
