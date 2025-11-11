from collections import defaultdict
import numpy as np
import random
import json

class Individual(object):
    def __init__(self):
        self.solution = []    
        self.objective = defaultdict()
 
        self.n = 0              
        self.rank = 0           
        self.S = []             
        self.distance = 0       
 
    def calculate_objective(self, objective_fun):
        self.objective = objective_fun(self.solution)
 
    def __lt__(self, other):
        v1 = list(self.objective.values())
        v2 = list(other.objective.values())
        if v1[0] < v2[0] or v1[1] < v2[1] or v1[2] < v2[2]:
            return 0
        return 1
 
def fast_non_dominated_sort(P):
    F = defaultdict(list)
 
    for p in P:
        p.S = []
        p.n = 0
        for q in P:
            if p < q:       
                p.S.append(q)
            elif q < p:    
                p.n += 1
        if p.n == 0:
            p.rank = 1
            F[1].append(p)
    i = 1
    while F[i]:
        Q = []
        for p in F[i]:
            for q in p.S:
                q.n -= 1
                if q.n == 0:
                    q.rank = i + 1
                    Q.append(q)
        i += 1
        F[i] = Q
 
    return F
 
def crowding_distance_assignment(L):
    l = len(L)
    for i in range(l):
        L[i].distance = 0
    for m in L[0].objective.keys():
        L.sort(key=lambda x: x.objective[m])    
        L[0].distance = float('inf')
        L[l - 1].distance = float('inf')
        f_max = L[l - 1].objective[m]
        f_min = L[0].objective[m]
        for i in range(1, l - 1):
            if f_max == f_min:
                f_max = f_min + 0.01
            L[i].distance = L[i].distance + (L[i + 1].objective[m] - L[i - 1].objective[m]) / (f_max - f_min)
 
def binary_tornament(ind1, ind2):
    if ind1.rank != ind2.rank:
        return ind1 if ind1.rank < ind2.rank else ind2
    elif ind1.distance != ind2.distance:
        return ind1 if ind1.distance > ind2.distance else ind2
    else:
        return ind1
 
def crossover_mutation(parent1, parent2, objective_fun, pc, pm):
    poplength = len(parent1.solution)   
    offspring1 = Individual()
    offspring2 = Individual()
  
    if random.random() <= pc:
        a = random.randint(0, poplength - 1)
        b = random.randint(0, poplength - 1)
        start, end = sorted([a, b])  
        segment_p1 = parent1.solution[start:end+1]
        remaining_p2 = [x for x in parent2.solution if x not in segment_p1]
        n_before = start
        n_after = poplength - (end + 1)
        offspring1.solution = remaining_p2[:n_before] + segment_p1 + remaining_p2[n_before:n_before + n_after]
        segment_p2 = parent2.solution[start:end+1]
        remaining_p1 = [x for x in parent1.solution if x not in segment_p2]
        offspring2.solution = remaining_p1[:n_before] + segment_p2 + remaining_p1[n_before:n_before + n_after]
    else:
        offspring1.solution = parent1.solution[:]  # 浅复制，确保父代的数据完全不受后代修改的影响
        offspring2.solution = parent2.solution[:]
        
    for offslt in [offspring1.solution, offspring2.solution]:
        for i in range(poplength):
            if random.random() <= pm:
                un_set = union_set - set(offslt)
                if un_set:
                    offslt[i] = random.choice(list(un_set)) 
                else:
                    if i > 0:
                        j = random.randint(0, i - 1)  
                        temp = offslt[j]      
                        offslt[j] = offslt[i]   
                        offslt[i] =  temp                             
                                       
    offspring1.calculate_objective(objective_fun)
    offspring2.calculate_objective(objective_fun)
    return [offspring1, offspring2]
 
def make_new_pop(P, objective_fun, pc, pm):
    popnum = len(P)    
    Q = []

    for i in range(int(popnum / 2)):
        i = random.randint(0, popnum - 1)
        j = random.randint(0, popnum - 1)
        parent1 = binary_tornament(P[i], P[j])    
        
        m = random.randint(0, popnum - 1)
        n = random.randint(0, popnum - 1)
        parent2 = binary_tornament(P[m], P[n])
        Two_offspring = crossover_mutation(parent1, parent2, objective_fun, pc, pm)
        Q.append(Two_offspring[0])
        Q.append(Two_offspring[1])
    return Q

def calc_prag(x, y):
    temp = 0
    sum = 0
    if len(y) == 0 or len(x) == 0 :
        return 0
    if len(x) == 1:
        if x == y:
            return 1
        else: 
            return 0
    for i, item_x1 in enumerate(x):
        for j, item_x2 in enumerate(x):
            if i >= j:
                continue
            id1 = -1
            id2 = -1
            for k, item_y in enumerate(y):
                if item_y == item_x1:
                    id1 = k
                if item_y == item_x2:
                    id2 = k
            sum = sum + 1
            if id1 == -1:
                continue
            if id2 == -1:
                temp = temp + 1
            if id1 < id2:
                temp = temp + 1

    return temp / sum

def calculate_precision(test_list, ground_truth):
    if not test_list: 
        return 0.0
    ground_truth_set = set(ground_truth)
    match_count = sum(1 for item in test_list if item in ground_truth_set)
    precision = match_count / len(test_list)

    return precision
 
def func1(x):
    f = defaultdict(float)
    sim = []
    sns = []
    for lists in categories.values():
        category_results = []
        for sublist in lists:
            value = calc_prag(sublist, x)
            category_results.append(value)
    
        sst_array = np.array(category_results)
        SNSR = sst_array.max() - sst_array.min()
        SNSV = sst_array.std()
        sim.append(sst_array.mean())
        sns.append(0.3 * SNSR + 0.7 * SNSV)
    f[1] = calculate_precision(x, ground_truth)              
    f[2] = 1 - np.array(sns).mean()
    f[3] = np.array(sim).mean() 
    return f
        
def main():
    generations = 250   
    popnum = 100     
    poplength = 10  
    objective_fun = func1
    pc = 0.9
    pm = 0.1
    
    neutral_ind = Individual()
    neutral_ind.objective = func1(neutral_list)
    
    P = []
    for i in range(popnum):
        P.append(Individual())
        P[i].solution = random.sample(union_list, poplength)
        P[i].calculate_objective(objective_fun) 
    
    F = fast_non_dominated_sort(P)
    Q = make_new_pop(P, objective_fun, pc, pm)  
    P_t = P   
    Q_t = Q     
    
    for gen_cur in range(generations):
        R_t = P_t + Q_t
        F = fast_non_dominated_sort(R_t)
        P_n = []    
        i = 1

        while len(P_n) + len(F[i]) < popnum:
            crowding_distance_assignment(F[i])
            P_n += F[i]
            i += 1

        crowding_distance_assignment(F[i])
        F[i].sort(key=lambda x: x.distance, reverse=True)
        P_n = P_n + F[i][:popnum - len(P_n)]
        Q_n = make_new_pop(P_n, objective_fun, pc, pm)
        
        P_t = P_n
        Q_t = Q_n
               
    count = 0
    f1_sum , f2_sum, f3_sum = 0, 0, 0
    for indv in F[1]:
        f1_sum += indv.objective[1]  
        f2_sum += indv.objective[2]  
        f3_sum += indv.objective[3]  
        count += 1
        
    print(f"user{user_id}")  
    print(f"    RecLLM     Accuracy:{neutral_ind.objective[1]}, Balance:{neutral_ind.objective[2]}, Correlation:{neutral_ind.objective[3]}")
    print(f"    RecLLMOEA  Accuracy:{f1_sum/count}, Balance:{f2_sum/count}, Correlation:{f3_sum/count}, num of Pareto Front:{count}")
     
    return neutral_ind.objective, f1_sum/count,  f2_sum/count, f3_sum/count, count           
 
if __name__ == "__main__":
    result_1 = {"Accuracy":0, "Balance":0, "Correlation":0}
    result_2 = {"Accuracy":0, "Balance":0, "Correlation":0, "num of Pareto Front":0}

    with open('', 'r', encoding='utf-8') as file:
        data = json.load(file)  
        
    n = 0
    for user in data:
        user_id = user['user_id']
        neutral_list = user['neutral_list']
        age_middle_list = user['age_middle_list']
        age_old_list = user['age_old_list']
        age_young_list = user['age_young_list']
        gender_male_list = user['gender_male_list']
        gender_female_list = user['gender_female_list']
        race_white_list = user['race_white_list']
        race_black_list = user['race_black_list']
        race_yellow_list = user['race_yellow_list']
        ground_truth = user['ground_truth']   
        

        all_lists = [age_middle_list, age_old_list, age_young_list, gender_male_list, gender_female_list, race_white_list, race_yellow_list, race_black_list]
        sets = [set(lst) for lst in all_lists]  
        union_set = set().union(*sets)
        union_list = list(union_set)
        
        categories = {
        "age": [age_middle_list, age_old_list, age_young_list],
        "gender": [gender_male_list, gender_female_list],
        "race": [race_white_list, race_yellow_list, race_black_list]
        }
        
        neu, acc, rel, fair, count = main()
        result_1['Accuracy'] += neu[1]
        result_1['Balance'] += neu[2]
        result_1['Correlation'] += neu[3]
        result_2['Accuracy'] += acc
        result_2['Balance'] += rel
        result_2['Correlation'] += fair
        result_2['num of Pareto Front'] += count
        
        n += 1
print()
print(f"The average results for all users are as follows:")
print(f"    RecLLM       Accuracy:{result_1['Accuracy']/n}, Balance:{result_1['Balance']/n},Correlation:{result_1['Correlation']/n}")
print(f"    RecLLMOEA    Accuracy:{result_2['Accuracy']/n}, Balance:{result_2['Balance']/n},Correlation:{result_2['Correlation']/n}, num of Pareto Front:{result_2['num of Pareto Front']/n}")
 
 

 
 