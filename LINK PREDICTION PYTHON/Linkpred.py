

 #``` python 
import networkx as nx 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 

# Load the Zachary Karate Club network dataset 
G = nx.karate_club_graph() # Generate a list of all possible edges in the network 
all_edges = list(nx.non_edges(G)) # Split the edges into training and testing sets

##### NOTE TODO: TEST ALL RESULTS WITHOUT random_state DEFINED ##### 
train_edges, test_edges = train_test_split(all_edges, test_size=0.3, random_state=42) 

# Define the link prediction algorithms 

def common_neighbors(G, u, v): 
    return len(list(nx.common_neighbors(G, u, v))) 
def jaccard_coefficient(G, u, v): 
    cn = common_neighbors(G, u, v) 
    return cn / len(set(G[u]) | set(G[v])) 
def preferential_attachment(G, u, v): 
    return len(G[u]) * len(G[v]) 

# Calculate scores for each algorithm on the training set 
cn_scores_train = [(u, v, common_neighbors(G, u, v)) for u, v in train_edges] 
jc_scores_train = [(u, v, jaccard_coefficient(G, u, v)) for u, v in train_edges] 
pa_scores_train = [(u, v, preferential_attachment(G, u, v)) for u, v in train_edges] 

# Sort the scores in descending order 
cn_scores_train.sort(key=lambda x: -x[2]) 
jc_scores_train.sort(key=lambda x: -x[2]) 
pa_scores_train.sort(key=lambda x: -x[2]) 

# Generate a list of true labels for the test set 
true_labels = [1 if G.has_edge(u, v) else 0 for u, v in test_edges] 

# Calculate scores for each algorithm on the test set 
cn_scores_test = [(u, v, common_neighbors(G, u, v)) for u, v in test_edges] 
jc_scores_test = [(u, v, jaccard_coefficient(G, u, v)) for u, v in test_edges] 
pa_scores_test = [(u, v, preferential_attachment(G, u, v)) for u, v in test_edges] 

# Sort the scores in descending order 
cn_scores_test.sort(key=lambda x: -x[2]) 
jc_scores_test.sort(key=lambda x: -x[2]) 
pa_scores_test.sort(key=lambda x: -x[2]) 

# Calculate accuracy scores for each algorithm using the Adamic-Adar index
cn_preds = [1 if cn_scores_test[i][2] > cn_scores_train[int(len(cn_scores_train)*0.1)][2] else 0 for i in range(len(cn_scores_test))]
jc_preds = [1 if jc_scores_test[i][2] > jc_scores_train[int(len(jc_scores_train)*0.1)][2] else 0 for i in range(len(jc_scores_test))]
pa_preds = [1 if pa_scores_test[i][2] > pa_scores_train[int(len(pa_scores_train)*0.1)][2] else 0 for i in range(len(pa_scores_test))]

cn_acc = accuracy_score(true_labels, cn_preds)
jc_acc = accuracy_score(true_labels, jc_preds)
pa_acc = accuracy_score(true_labels, pa_preds)

# Print the accuracy scores 
print("Common Neighbors Accuracy:", cn_acc)
print("Jaccard's Coefficient Accuracy:", jc_acc)
print("Preferential Attachment Accuracy:", pa_acc)

#``` This code generates all possible edges in the Zachary Karate Club network dataset, splits them into training and testing sets, defines the link prediction algorithms, calculates the scores for each algorithm on the training and testing sets, sorts the scores in descending order, generates true labels for the test set, calculates accuracy scores for each algorithm using the Adamic-Adar index, and prints the accuracy scores. You can modify the code to use different datasets and link prediction algorithms as needed.
