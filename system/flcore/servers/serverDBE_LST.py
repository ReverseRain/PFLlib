from flcore.servers.serverbase import Server
from flcore.clients.clientDBE_LST import clientDBE_LST
import time
import copy

class FedDBE_LST(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        

        self.set_slow_clients()
        self.set_clients(clientDBE_LST)
        self.selected_clients = self.clients
        for client in self.selected_clients:
            client.train() # no DBE

        # Initialize global mean
        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
            
        global_mean = sum(client.running_mean * w for client, w in zip(self.selected_clients, self.uploaded_weights))
        print('>>>> global_mean <<<<', global_mean)
        for client in self.selected_clients:
            client.global_mean = global_mean.clone()
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        
        self.Budget = []
        print('featrue map shape: ', self.clients[0].client_mean.shape)
        print('featrue map numel: ', self.clients[0].client_mean.numel())

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\nRound {i}: Evaluate Model")
                self.evaluate()

            # Train selected clients
            for client in self.selected_clients:
                client.train()

            # Aggregate models
            self.receive_models()
            self.aggregate_parameters()

            # Budget tracking
            time_cost = time.time() - s_t
            self.Budget.append(time_cost)
            print(f"Time cost for round {i}: {time_cost:.2f}s")

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # Final evaluation and save results
        print("\nBest accuracy achieved:", max(self.rs_test_acc))
        avg_time_cost = sum(self.Budget[1:]) / len(self.Budget[1:])
        print(f"Average time cost per round: {avg_time_cost:.2f}s")

        self.save_results()
        self.save_global_model()
        
    # def aggregate_parameters(self):
    #     """
    #     Aggregate parameters from the uploaded models (backbone and LST).
    #     """
    #     assert len(self.uploaded_models) > 0

    #     # Initialize the global model's parameters to zero
    #     self.global_model = copy.deepcopy(self.uploaded_models[0])
    #     for param in self.global_model.parameters():
    #         param.data.zero_()
    
    #     # Aggregate parameters from all uploaded models
    #     for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
    #         self.add_parameters(w, client_model)
            
    # def add_parameters(self, w, client_model):
    #     """
    #     Add parameters from a single client model to the global model.
    #     Includes both backbone and LST module parameters.
    #     """
    #     # Iterate over backbone parameters
    #     for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
    #         server_param.data += client_param.data.clone() * w

    #     # Aggregate LST parameters separately if present
    #     if hasattr(client_model, 'ladder_module'):
    #         for server_param, client_param in zip(
    #                 self.global_model.ladder_module.parameters(),
    #                 client_model.ladder_module.parameters()):
    #             server_param.data += client_param.data.clone() * w
    


