import torch
import torch.nn as nn
from torch.autograd import Variable

USE_CUDA = False #torch.cuda.is_available()

class DMN(nn.Module):
    def __init__(self, config):
        super(DMN, self).__init__()
        self.hidden_size = config['hidden_size']
        self.embed = nn.Embedding(config['input_size'], self.hidden_size, padding_idx=0)
        self.context_gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.question_gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size * 7, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 2),
            nn.Softmax()
        )
        self.attention_grucell = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.memory_grucell = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.answer_gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(config['dropout_p'])
        self.num_episode = config['num_episode']

    def init_hidden(self, inputs):
        hidden = Variable(torch.zeros(1, inputs.size(0), self.hidden_size))
        return hidden.cuda() if USE_CUDA else hidden

    def init_weight(self):
        nn.init.xavier_uniform(self.embed.state_dict()['weight'])

        for name, param in self.context_gru.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.question_gru.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.gate.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.attention_grucell.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.memory_grucell.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.answer_gru.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)

    def forward(self, c_p, c_mask, q_p, q_mask, a_I_p, a_I_mask, a_II_p, a_II_mask, is_training=False):
        C = []
        for c, c_m in zip(c_p, c_mask):
            embeds = self.embed(c)
            if is_training:
                embeds = self.dropout(embeds)
            hidden = self.init_hidden(c)
            outputs, hidden = self.context_gru(embeds, hidden)
            real_hidden = []

            for i, o in enumerate(outputs):
                real_length = c_m[i].data.tolist().count(0)
                real_hidden.append(o[real_length - 1])

            C.append(torch.cat(real_hidden).view(c.size(0), -1).unsqueeze(0))
        encoded_context = torch.cat(C)

        # Question Module
        embeds = self.embed(q_p)
        if is_training:
            embeds = self.dropout(embeds)
        hidden = self.init_hidden(q_p)
        outputs, hidden = self.question_gru(embeds, hidden)

        if isinstance(q_mask, torch.autograd.variable.Variable):
            real_question = []
            for i, o in enumerate(outputs):
                real_length = q_mask[i].data.tolist().count(0)
                real_question.append(o[real_length - 1])
            encoded_question = torch.cat(real_question).view(q_p.size(0), -1)
        else:  # during testing
            encoded_question = hidden.squeeze(0)

        # Answer Module I
        embeds = self.embed(a_I_p)
        hidden = self.init_hidden(a_I_p)
        outputs, hidden = self.answer_gru(embeds, hidden)

        if isinstance(a_I_mask, torch.autograd.variable.Variable):
            real_answer = []
            for i, o in enumerate(outputs):
                real_length = a_I_mask[i].data.tolist().count(0)
                real_answer.append(o[real_length - 1])
            encoded_answer_c = torch.cat(real_answer).view(a_I_p.size(0), -1)
        else:  # during testing
            encoded_answer_c = hidden.squeeze(0)

        # Answer Module II
        embeds = self.embed(a_II_p)
        hidden = self.init_hidden(a_II_p)
        outputs, hidden = self.answer_gru(embeds, hidden)

        if isinstance(a_II_mask, torch.autograd.variable.Variable):
            real_answer = []
            for i, o in enumerate(outputs):
                real_length = a_II_mask[i].data.tolist().count(0)
                real_answer.append(o[real_length - 1])
            encoded_answer_i = torch.cat(real_answer).view(a_II_p.size(0), -1)
        else:  # during testing
            encoded_answer_i = hidden.squeeze(0)

        # Episodic Memory Module
        memory = encoded_question
        T_C = encoded_context.size(1)
        B = encoded_context.size(0)
        for i in range(self.num_episode):
            hidden = self.init_hidden(encoded_context.transpose(0, 1)[0]).squeeze(0)
            for t in range(T_C):
                z = torch.cat([encoded_context.transpose(0, 1)[t], memory, encoded_question,
                    encoded_context.transpose(0, 1)[t] * encoded_question,  # element-wise product
                    encoded_context.transpose(0, 1)[t] * memory,  # element-wise product
                    torch.abs(encoded_context.transpose(0, 1)[t] - encoded_question),
                    torch.abs(encoded_context.transpose(0, 1)[t] - memory)
                ], 1)
                g_t = self.gate(z)
                hidden = g_t * self.attention_grucell(encoded_context.transpose(0, 1)[t], hidden) + (1 - g_t) * hidden
            e = hidden
            memory = self.memory_grucell(e, memory)
        return self.output_layer(torch.cat([memory, encoded_answer_c, encoded_answer_i], 1))

