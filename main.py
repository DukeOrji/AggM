#main.py
from ds import load_cifar

from agg_methods import Aggregation
from user import User
from config import device, SAVE

import pandas as pd
import os


results = []
num_clients = 50
user_dataloader, test_loader = load_cifar(num_clients)


users = [User(i, user_dataloader[i]) for i in range(num_clients)]

server = Aggregation()


rng_num = 50
print(next(server.global_model.parameters()).device) #print gpu or cpu
for epoch in range(rng_num):
    print(f"\nRound: {epoch+1}")
    #server broadcast weight
    global_weight = server.broadcast_weight()

    #clients train on global weigth
    for user in users:
        user.set_weight(global_weight)
        loss, acc = user.train()
        print(f"Client: {user.user_id}  |  Loss: {loss}  |  Acc: {acc}")


    user_weights = [
        user.get_weight()
        for user in users
    ]

    server.FedMed(user_weights)
    global_loss, global_acc = server.evaluate(test_loader)
    print(f"\n Cloud Loss: {global_loss}  |  Cloud Acc: {global_acc}")

    results.append({
        "Round": epoch + 1,
        "Global Loss": global_loss,
        "Global Accuracy": global_acc
    })


    
    if epoch < rng_num-1:
        print("Next Round ...")
    else:
        print("Experiment Complete.")    


if SAVE:

    # attack = ""
    # defense = ""
    # save_dir = f"feedback/{defense}/{attack}"
    save_dir = "feedback/FedMedian"

    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(
        f"{save_dir}/results.csv",
        index=False
    )   

