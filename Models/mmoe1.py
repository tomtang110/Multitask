# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config(object):
    def __init__(self,data_dir):
        self.model_name = 'mmoe'
        self.train_path = data_dir + 'census-income.data.gz'
        self.test_path = data_dir + 'census-income.test.gz'
        self.save_path = './saved_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_improvement = 1000
        self.dropout = 0.5
        self.learning_rate = 1e-4
        self.label_columns = ['income_50k', 'marital_stat']

        self.label_dict = [2,2]
        self.num_feature = 0
        self.num_experts = 6
        self.num_tasks = 2
        self.units= 16
        self.hidden_units= 8

        self.embed_size = 64
        self.batch_size = 512
        self.expert_hidden = 32
        self.field_size = 0

        self.num_epochs = 50


class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        # self.log_soft = nn.LogSoftmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        # out = self.log_soft(out)
        return out

class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)
        # out = torch.sigmoid(out)
        return out


class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # params
        self.input_size = config.num_feature
        self.num_experts = config.num_experts
        self.experts_out = config.units
        self.experts_hidden = config.expert_hidden
        self.towers_hidden = config.hidden_units
        self.tasks = len(config.label_dict)
        # row by row
        self.softmax = nn.Softmax(dim=1)
        # model
        self.experts = nn.ModuleList(
            [Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_experts)])
        self.w_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(self.input_size, self.num_experts), requires_grad=True) for i in range(self.tasks)])
        self.towers = nn.ModuleList([Tower(self.experts_out, 1, self.towers_hidden) for i in range(self.tasks)])

    def forward(self, x):
        # get the experts output
        expers_o = [e(x) for e in self.experts]
        expers_o_tensor = torch.stack(expers_o)

        # get the gates output
        gates_o = [self.softmax(x @ g) for g in self.w_gates]

        # multiply the output of the experts with the corresponding gates output
        # res = gates_o[0].t().unsqueeze(2).expand(-1, -1, self.experts_out) * expers_o_tensor
        # https://discuss.pytorch.org/t/element-wise-multiplication-of-the-last-dimension/79534
        towers_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * expers_o_tensor for g in gates_o]
        towers_input = [torch.sum(ti, dim=0) for ti in towers_input]

        # get the final output from the towers
        final_output = [t(ti) for t, ti in zip(self.towers, towers_input)]

        # get the output of the towers, and stack them
        # final_output = torch.stack(final_output, dim=1)

        return final_output
