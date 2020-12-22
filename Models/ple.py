# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import loss_fn
class Config(object):
    def __init__(self,data_dir):
        self.model_name = 'mmoe'
        self.train_path = data_dir + 'census-income.data.gz'
        self.test_path = data_dir + 'census-income.test.gz'
        self.save_path = './saved_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_improvement = 40000
        self.dropout = 0.5
        self.learning_rate = 1e-3
        self.label_columns = ['income_50k', 'marital_stat']

        self.label_dict = [2,2]
        self.num_feature = 0
        self.num_experts = 8
        self.num_tasks = 2
        self.units= 16
        self.hidden_units= 8

        self.embed_size = 300
        self.batch_size = 2048

        self.field_size = 0

        self.towers_hidden = 16

        self.num_epochs = 100
        self.loss_fn = loss_fn('binary')
        self.selectors = 2

class Expert_shared(nn.Module):
    def __init__(self,input_shape,output_shape):
        super(Expert_shared,self).__init__()

        self.fc1 = nn.Linear(input_shape,output_shape)
    def forward(self,x):
        return self.fc1(x)

class Expert_task1(nn.Module):
    def __init__(self,input_shape,output_shape):
        super(Expert_task1,self).__init__()

        self.fc1 = nn.Linear(input_shape,output_shape)
    def forward(self,x):
        return self.fc1(x)

class Expert_task2(nn.Module):
    def __init__(self,input_shape,output_shape):
        super(Expert_task2,self).__init__()

        self.fc1 = nn.Linear(input_shape,output_shape)
    def forward(self,x):
        return self.fc1(x)

class Gate_shared(nn.Module):
    def __init__(self,input_shape,output_shape):
        super(Gate_shared,self).__init__()
        self.fc1 = nn.Linear(input_shape,output_shape)
    def forward(self,x):
        return self.fc1(x)


class Gate_task1(nn.Module):
    def __init__(self,input_shape,output_shape):
        super(Gate_task1,self).__init__()
        self.fc1 = nn.Linear(input_shape,output_shape)
    def forward(self,x):
        return self.fc1(x)

class Gate_task2(nn.Module):
    def __init__(self,input_shape,output_shape):
        super(Gate_task2,self).__init__()
        self.fc1 = nn.Linear(input_shape,output_shape)
    def forward(self,x):
        return self.fc1(x)

class GatingNetwork(nn.Module):

    def __init__(self,input_units,units,num_experts,selectors):
        super(GatingNetwork,self).__init__()

        self.experts_shared = nn.ModuleList([Expert_shared(input_units,units)
                                     for i in range(num_experts)])
        self.experts_task1 = nn.ModuleList([Expert_task1(input_units,units)
                                     for i in range(num_experts)])
        self.experts_task2 = nn.ModuleList([Expert_task2(input_units, units)
                                            for i in range(num_experts)])
        self.expert_activation = nn.ReLU()

        self.gate_shared = Gate_shared(input_units,num_experts*3)
        self.gate_task1 = Gate_task1(input_units,selectors*num_experts)
        self.gate_task2 = Gate_task2(input_units, selectors*num_experts)

        self.gate_activation = nn.Softmax(dim=-1)
        self.units = units
        self.num_expers = num_experts
    def forward(self,gate_output_shared_final,gate_output_task1_final,gate_output_task2_final):

        # expert shared
        expert_shared_o = [e(gate_output_shared_final) for e in self.experts_shared]
        expert_shared_tensors = torch.cat(expert_shared_o,dim=0)
        expert_shared_tensors = expert_shared_tensors.view(-1,self.num_expers,self.units)
        expert_shared_tensors = self.expert_activation(expert_shared_tensors)
        # expert task1
        expert_task1_o = [e(gate_output_task1_final) for e in self.experts_task1]
        expert_task1_tensors = torch.cat(expert_task1_o, dim=0)
        expert_task1_tensors = expert_task1_tensors.view(-1,self.num_expers, self.units)
        expert_task1_tensors = self.expert_activation(expert_task1_tensors)
        # expert task2
        expert_task2_o = [e(gate_output_task2_final) for e in self.experts_task2]
        expert_task2_tensors = torch.cat(expert_task2_o, dim=0)
        expert_task2_tensors = expert_task2_tensors.view(-1,self.num_expers, self.units)
        expert_task2_tensors = self.expert_activation(expert_task2_tensors)
        # gate task1
        gate_output_task1 = self.gate_task1(gate_output_task1_final)
        gate_output_task1 = self.gate_activation(gate_output_task1)

        gate_expert_output1 = torch.cat([expert_shared_tensors,expert_task1_tensors],dim=1)

        gate_output_task1 = torch.einsum('be,beu ->beu',gate_output_task1,gate_expert_output1)
        gate_output_task1 = gate_output_task1.sum(dim=1)
        # gate task2
        gate_output_task2 = self.gate_task2(gate_output_task2_final)
        gate_output_task2 = self.gate_activation(gate_output_task2)

        gate_expert_output2 = torch.cat([expert_shared_tensors, expert_task2_tensors], dim=1)

        gate_output_task2 = torch.einsum('be,beu ->beu', gate_output_task2, gate_expert_output2)
        gate_output_task2 = gate_output_task2.sum(dim=1)
        # gate shared
        gate_output_shared = self.gate_shared(gate_output_shared_final)
        gate_output_shared  = self.gate_activation(gate_output_shared )

        gate_expert_output_shared = torch.cat([expert_task1_tensors,expert_shared_tensors, expert_task2_tensors], dim=1)

        gate_output_shared = torch.einsum('be,beu ->beu', gate_output_shared, gate_expert_output_shared)
        gate_output_shared = gate_output_shared.sum(dim=1)

        return gate_output_shared,gate_output_task1,gate_output_task2


class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        out = torch.sigmoid(out)
        return out

class Model(nn.Module):

    def __init__(self,config):
        super(Model,self).__init__()

        self.gate1 = GatingNetwork(config.num_feature,config.units,config.num_experts,config.selectors)
        self.gate2 = GatingNetwork(config.units, config.units, config.num_experts, config.selectors)

        self.towers = nn.ModuleList([
            Tower(config.units, 1, config.towers_hidden) for i in range(config.num_tasks)
        ])
    def forward(self,x):
        gate_output_shared, gate_output_task1, gate_output_task2 = self.gate1(x,x,x)
        _,task1_o,task2_o = self.gate2(gate_output_shared, gate_output_task1, gate_output_task2)

        final_output = [tower(task) for tower,task in zip(self.towers,[task1_o,task2_o])]

        return final_output



