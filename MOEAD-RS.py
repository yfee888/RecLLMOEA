from collections import defaultdict
import numpy as np
import random
import json

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
    f = []
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
    f = [calculate_precision(x, ground_truth), 1 - np.array(sns).mean(), np.array(sim).mean()]
    return f

class Individual():
    def __init__(self):
        self.solution = []
        self.f = [] 
    
    def __lt__(self, other):   
        for i in range(len(self.f)):
            if self.f[i] < other.f[i]:
                return 0
        return 1
    
    def calculate_objective(self):
        self.f = func1(self.solution)
    
def generate_lists(num_lists=100):
    result = []

    for i in range(num_lists):
        b = i * 0.01
        if b <= 1:
            a = np.random.uniform(0, 1 - b)  
            c = 1 - a - b 
            result.append([a, b, c])

    return result
                   
def initial(N, poplength):
    P=[]

    for i in range(N):
        P.append(Individual())
        P[i].solution = random.sample(union_list, poplength)
        P[i].calculate_objective() 
        lamda = generate_lists()

    return P, lamda

def look_neighbor(lamda,T):
    B=[]
    for i in range(len(lamda)):
        temp=[]
        for j in range(len(lamda)):
            distance=np.sqrt((lamda[i][0]-lamda[j][0])**2+(lamda[i][1]-lamda[j][1])**2)
            temp.append(distance)
        l=np.argsort(temp)
        B.append(l[:T])
    return B

def bestvalue(P):
    best=[]
    for i in range(len(P[0].f)):
        best.append(P[0].f[i])
    for i in range(1,len(P)):
        for j in range(len(P[i].f)):
            if P[i].f[j]<best[j]:
                best[j]=P[i].f[j]
    return best

def crossover_mutation(parent1, parent2, pc, pm):

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
        offspring1.solution = parent1.solution[:]  
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
                                       
    offspring1.calculate_objective()
    offspring2.calculate_objective()
    return [offspring1, offspring2]
         
def Tchebycheff(x,lamb,z):
    temp=[]
    for i in range(len(x.f)):
        temp.append(np.abs(x.f[i]-z[i])*lamb[i])
    return np.max(temp)

def ws(x,lamda):
    temp=0.0
    for i in range(len(x.f)):
        temp=temp+float(x.f[i]*lamda[i])
    return temp

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

def main():  
    N = 100 
    T = 10 
    poplenth = 10
    max_gen = 250 
    pc = 0.9
    pm = 0.1

    neutral_ind = Individual()
    neutral_ind.f = func1(neutral_list)

    P,lamda=initial(N, poplenth)
    B=look_neighbor(lamda,T)
    z=bestvalue(P)

    gen=1
    while(gen < max_gen):
        for i in range(N):
            k=random.randint(0,T-1)
            l=random.randint(0,T-1)
            y1, y2 = crossover_mutation(P[B[i][k]], P[B[i][l]], pc, pm)
            if y1 < y2:
                y=y1
            else:
                y=y2

            for j in range(len(z)):
                if y.f[j] > z[j]:
                    z[j] = y.f[j]

            for j in range(len(B[i])):
                gte_xi=ws(P[B[i][j]], lamda[B[i][j]])
                gte_y=ws(y, lamda[B[i][j]])
                if (gte_y >= gte_xi):
                    P[B[i][j]]=y
                         
        gen=gen+1
    
    F = fast_non_dominated_sort(P)
    count = 0
    f1_sum , f2_sum, f3_sum = 0, 0, 0
    for indv in F[1]:
        f1_sum += indv.f[0]  
        f2_sum += indv.f[1]  
        f3_sum += indv.f[2]  
        count += 1
    
    print(f"user{user_id}")  
    print(f"    RecLLM      Accuracy:{neutral_ind.f[0]}, Balance:{neutral_ind.f[1]}, Correlation:{neutral_ind.f[2]}")
    print(f"    RecLLMOEA   Accuracy:{f1_sum/count}, Balance:{f2_sum/count}, Correlation:{f3_sum/count}, num of Pareto Front:{count}")
    return neutral_ind.f, f1_sum/count,  f2_sum/count, f3_sum/count, count

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
        result_1['Accuracy'] += neu[0]
        result_1['Balance'] += neu[1]
        result_1['Correlation'] += neu[2]
        result_2['Accuracy'] += acc
        result_2['Balance'] += rel
        result_2['Correlation'] += fair
        result_2['num of Pareto Front'] += count
        
        n += 1
print()
print(f"The average results for all users are as follows:")
print(f"    RecLLM       Accuracy:{result_1['Accuracy']/n}, Balance:{result_1['Balance']/n},Correlation:{result_1['Correlation']/n}")
print(f"    RecLLMOEA    Accuracy:{result_2['Accuracy']/n}, Balance:{result_2['Balance']/n},Correlation:{result_2['Correlation']/n}, num of Pareto Front:{result_2['num of Pareto Front']/n}")
 