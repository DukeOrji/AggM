from server import Server  
import torch
import copy
from statistics import mean
 
    
  
    
class Aggregation(Server):
    
    def FedMed(self, client_weights):
        median_weights = {}

        for key in client_weights[0]:
            if client_weights[0][key].dtype.is_floating_point:
                med = torch.zeros_like(client_weights[0][key])
                
                param = []
                for weight in client_weights:
                    param.append(weight[key])

                stacked = torch.stack(param)
                median_weights[key] = torch.median(stacked, dim=0).values

            else:
                median_weights[key] = client_weights[0][key].clone()

        self.global_model.load_state_dict(median_weights)
                

    def Krum(self, client_weights, num_attackers=0):
        num_clients = len(client_weights)
        m = num_clients - num_attackers - 2
        scores = []
        

        for i in range(num_clients):
            distances = []
            for j in range(num_clients):

                if i == j:
                    continue

                dist = 0.0
                for key in client_weights[i].keys():
                    if client_weights[i][key].dtype.is_floating_point:

                        diff = client_weights[i][key] - client_weights[j][key]
                        dist += diff.pow(2).sum().item()
                distances.append(dist)

           
            distances.sort()
            scores.append(sum(distances[:m]))
        
        winner = scores.index(min(scores))

        print(
            f"\nKrum selected client {winner} "
            f"(score={scores[winner]:.2f})"
        )

        self.global_model.load_state_dict(copy.deepcopy(client_weights[winner]))

        
        
                

    def FedAvg(self, client_weights):

        avg_weights = {}

        for key in client_weights[0]:

            if client_weights[0][key].dtype.is_floating_point:

                avg = torch.zeros_like(client_weights[0][key])

                for weights in client_weights:
                    avg += weights[key]

                avg /= len(client_weights)

                avg_weights[key] = avg

            else:

                avg_weights[key] = client_weights[0][key].clone()

        self.global_model.load_state_dict(avg_weights)


    def set_weight(self, weight):

        self.global_model.load_state_dict(weight)