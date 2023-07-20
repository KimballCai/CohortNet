import torch
from torch import nn as nn
import numpy as np
import copy

def random_init(dataset, num_centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)

    indices = torch.tensor(np.random.randint(0, num_points, int(num_centers)), dtype=torch.long).cuda()
    centers = torch.gather(dataset, 0, indices.view(-1, 1).expand(-1, dimension))
    return centers


def compute_codes(dataset, centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    num_centers = centers.size(0)
    chunk_size = int(1e8 / num_centers)
    codes = torch.zeros(num_points, dtype=torch.long).cuda()
    centers_t = torch.transpose(centers, 0, 1)
    centers_norms = torch.sum(centers ** 2, dim=1).view(1, -1)
    for i in range(0, num_points, chunk_size):
        begin = i
        end = min(begin + chunk_size, num_points)
        dataset_piece = dataset[begin:end, :]
        dataset_norms = torch.sum(dataset_piece ** 2, dim=1).view(-1, 1)
        distances = torch.mm(dataset_piece, centers_t)
        # (x-y)^2 = x^2 - 2*xy + y^2
        distances *= -2.0
        distances += dataset_norms
        distances += centers_norms
        _, min_ind = torch.min(distances, dim=1)
        codes[begin:end] = min_ind
    return codes


def update_centers(dataset, codes, num_centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    centers = torch.zeros(num_centers, dimension, dtype=torch.float).cuda()
    cnt = torch.zeros(num_centers, dtype=torch.float).cuda()
    centers.scatter_add_(0, codes.view(-1, 1).expand(-1, dimension), dataset)
    cnt.scatter_add_(0, codes, torch.ones(num_points, dtype=torch.float).cuda())
    cnt = torch.where(cnt > 0.5, cnt, torch.ones(num_centers, dtype=torch.float).cuda())
    centers /= cnt.view(-1, 1)
    return centers


def cluster(dataset, num_centers, centers):
    if centers == None:
        centers = random_init(dataset, num_centers)
    else:
        assert len(centers) == num_centers
    codes = compute_codes(dataset, centers)
    num_iterations = 0
    while True:
        centers = update_centers(dataset, codes, num_centers)
        new_codes = compute_codes(dataset, centers)
        # converge
        if torch.equal(codes, new_codes):
            break
        if num_iterations > 1000:
            break
        codes = new_codes
        num_iterations += 1
    return centers, codes


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class BiEmbedding(nn.Module):
    # Bi-directional Embedding Module in ELDA
    def __init__(self, f_dim, e_dim, clip_max, clip_min):
        super(BiEmbedding, self).__init__()

        # dimensions
        self.f_dim = f_dim  # feature dimension
        self.e_dim = e_dim  # embedding dimension
        self.clip_max = clip_max  # clip_max for features
        self.clip_min = clip_min  # clip_min for features

        # models & parameters
        self.embed0 = nn.Parameter(torch.ones(self.f_dim, self.e_dim), requires_grad=True)
        self.embed1 = nn.Parameter(torch.ones(self.f_dim, self.e_dim), requires_grad=True)
        self.embedm = nn.Parameter(torch.ones(self.f_dim, self.e_dim), requires_grad=True)

        self.embed0 = nn.init.kaiming_normal_(self.embed0, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.embed1 = nn.init.kaiming_normal_(self.embed1, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.embedm = nn.init.kaiming_normal_(self.embedm, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, tdata, tmask):
        assert tdata.shape == tmask.shape, "[x] The shape of tdata is not same as the shape of tmask."
        batch_size, t_dim, f_dim = tdata.shape

        f_mask = torch.sum(tmask, -2)
        f_mask = f_mask.bool().float()
        f_mask = f_mask.unsqueeze(-2)
        f_mask = f_mask.repeat([1, t_dim, 1])  # B * T * F

        e0 = (tdata - self.clip_min).unsqueeze(-1) * self.embed0
        e1 = (self.clip_max - tdata).unsqueeze(-1) * self.embed1
        em = (1 - f_mask).unsqueeze(-1) * self.embedm
        embed = f_mask.unsqueeze(-1) * (e0 + e1) / (self.clip_max - self.clip_min) + em

        # return embed  # B * T * F * E
        return embed, f_mask  # B * T * F * E; # B * T * F

class FeatureInteractionLearning(nn.Module):
    # Feature-level Interaction Learning Module in ELDA_net
    def __init__(self, f_dim, e_dim, active='relu', drop_f=0.3, inter_type="mul", speed_up=True):
        super(FeatureInteractionLearning, self).__init__()
        self.f_dim = f_dim  # feature dimension
        self.e_dim = e_dim  # embedding dimension
        self.inter_type = inter_type  # interaction type: mul or add
        self.speed_up = speed_up

        self.a_ff = clones(nn.Linear(self.e_dim, self.e_dim), self.f_dim)
        self.a_ff2 = clones(nn.Linear(self.e_dim, 1, bias=False), self.f_dim)

        if active == "relu":
            self.active = nn.ReLU(inplace=True)
        elif active == 'dropout':
            self.active = nn.Dropout(drop_f, inplace=True)

        self.softmax = nn.Softmax(2)

    def forward(self, in_embeds, f_mask):  # B * T * F * E; B * T * F
        batch_size, t_dim, f_dim, e_dim = in_embeds.shape
        assert f_dim == self.f_dim, "[x] Not match: f_dim=%d & self.f_dim=%d" % (f_dim, self.f_dim)
        assert e_dim == self.e_dim, "[x] Not match: e_dim=%d & self.e_dim=%d" % (e_dim, self.e_dim)

        a_masks = 10e6 * torch.eye(self.f_dim).cuda()  # F * F
        all_ff_out, all_ff_a, all_f_rep = None, None, None
        all_cat_out, all_compress = None, None
        for i in range(self.f_dim):
            org_embed = in_embeds[:, :, i, :].unsqueeze(2)  # B * T * 1 * E
            oth_embed = in_embeds  # B * T * F * E
            if self.inter_type == 'mul':
                a_in = org_embed * oth_embed  # B * T * F * E
            elif self.inter_type == 'add':
                a_in = org_embed + oth_embed  # B * T * F * E

            a_out = self.a_ff[i](a_in)  # B * T * F * E
            a_out = self.a_ff2[i](self.active(a_out))  # B * T * F * 1
            a_out = a_out - a_masks[i].unsqueeze(-1)  # B * T * F * 1
            # disregard the missing feature when calculate interaction
            a_out = a_out - 10e6 * (1-f_mask).unsqueeze(-1)  # B * T * F * 1
            a_score = self.softmax(a_out)  # B * T * F * 1

            ff_out = (a_in * a_score).sum(-2)  # B * T * E

            if i == 0:
                all_ff_out = ff_out.unsqueeze(2)  # B * T * 1 * E
                all_ff_a = a_score.unsqueeze(2)  # B * T * 1 * F * 1
            else:
                all_ff_out = torch.cat([all_ff_out, ff_out.unsqueeze(2)], 2)  # B * T * F * E
                all_ff_a = torch.cat([all_ff_a, a_score.unsqueeze(2)], 2)  # B * T * F * F * 1
        return all_ff_out, all_ff_a.squeeze()

class FeatureTrendLearning(nn.Module):
    # to learn feature trend in each feature
    def __init__(self, e_dim, f_num):
        super(FeatureTrendLearning, self).__init__()
        self.f_num = f_num
        # local GRU
        self.embedRnns = clones(nn.GRU(e_dim, e_dim, batch_first=True), f_num)

    def forward(self, embed):
        # add temporal attribute to embed
        self.embedRnns[0].flatten_parameters()
        t_embed, _ = self.embedRnns[0](embed[:, :, 0, :])  # B * T * E
        t_embed = t_embed.unsqueeze(2)  # B * T * 1 * E
        for i in range(1, self.f_num):
            self.embedRnns[i].flatten_parameters()
            sub_t_embed, _ = self.embedRnns[i](embed[:, :, i, :])
            t_embed = torch.cat((t_embed, sub_t_embed.unsqueeze(2)), 2)  # B * T * F * E
        return t_embed # B * T * F * E

class FeatureFusion(nn.Module):
    def __init__(self, in_dim, hidden_dim, active="relu"):
        super(FeatureFusion, self).__init__()
        if active == "relu":
            self.active = nn.ReLU()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.compressor = nn.Linear(in_dim, hidden_dim)

    def forward(self, data):
        compress_out = self.compressor(data)  # B * T * F * C
        compress_out = self.active(compress_out)
        return compress_out  # B * T * F * C

class MultiChannelFeatureLearningModule(nn.Module):
    # to learn each feature in a separate channel
    def __init__(self, f_num, e_dim, clip_max, clip_min, fusion_dim, h_dim, c_dim, o_dim, active='relu'):
        super(MultiChannelFeatureLearningModule, self).__init__()
        self.f_num = f_num

        self.biEmbedding = BiEmbedding(f_num, e_dim, clip_max, clip_min)
        self.featureTrend = FeatureTrendLearning(e_dim, f_num)
        self.featureInteractionModule = FeatureInteractionLearning(f_num, e_dim, active)
        self.featureFusionModule = FeatureFusion(3 * e_dim, fusion_dim)

        # global GRU
        self.rnns = clones(nn.GRU(fusion_dim, h_dim, batch_first=True), f_num)
        for i in range(f_num):
            torch.nn.init.xavier_uniform_(self.rnns[i].weight_ih_l0)
            torch.nn.init.xavier_uniform_(self.rnns[i].weight_hh_l0)

        if active == 'relu':
            self.active = nn.ReLU()

        self.linears = clones(nn.Linear(h_dim, c_dim), f_num)
        self.prediction = nn.Linear(c_dim * f_num, o_dim)

    def forward(self, tdata, tmask):
        embed, f_mask = self.biEmbedding(tdata, tmask)  # B * T * F * E; # B * T * F

        # to learn feature trend
        t_embed = self.featureTrend(embed)
        # to learn feature interaction
        f_inters, ff_a = self.featureInteractionModule(embed, f_mask)  # B * T * F * E;  B * T * F * F

        used_embed = (1 - f_mask).unsqueeze(-1) * embed + f_mask.unsqueeze(-1) * t_embed
        fusion_in = torch.cat((embed, used_embed, f_inters), dim=-1)  # B * T * F * 3E
        # compress feature
        f_out = self.featureFusionModule(fusion_in)  # B * T * F * Cp

        # feature learning
        i = 0
        self.rnns[i].flatten_parameters()
        t_out, _ = self.rnns[i](f_out[:, :, i, :])  # B * T * H
        ct_out = self.active(t_out)
        # feature compression
        ct_out = self.linears[i](ct_out)  # B * T * Cp
        t_out = t_out.unsqueeze(-2)  # B * T * 1 * H
        ct_out = ct_out.unsqueeze(-2)  # B * T * 1 * Cp

        for i in range(1, self.f_num):
            self.rnns[i].flatten_parameters()
            sub_t_out, n = self.rnns[i](f_out[:, :, i, :])  # B * T * H
            sub_ct_out = self.active(sub_t_out)
            sub_ct_out = self.linears[i](sub_ct_out)  # B * T * C
            t_out = torch.cat((t_out, sub_t_out.unsqueeze(-2)), 2)  # B * T * F * H
            ct_out = torch.cat((ct_out, sub_ct_out.unsqueeze(-2)), 2)  # B * T * F * Cp

        pre_out = self.prediction(torch.flatten(ct_out[:, -1, :, :], start_dim=1, end_dim=2))
        return pre_out, t_out, f_mask, ff_a


class FinalAttentionQKV(nn.Module):
    def __init__(self, hidden_dim, query_dim, key_dim, value_dim=None, attention_type='mul', dropout=None):
        super(FinalAttentionQKV, self).__init__()

        self.attention_type = attention_type
        self.hidden_dim = hidden_dim
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.W_q = nn.Linear(query_dim, hidden_dim)
        self.W_k = nn.Linear(key_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.W_q.weight, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.W_k.weight, a=np.sqrt(5))

        if value_dim is not None:
            self.W_v = nn.Linear(key_dim, value_dim)
            nn.init.kaiming_uniform_(self.W_v.weight, a=np.sqrt(5))

        if attention_type == 'add':
            self.W_out = nn.Linear(hidden_dim, 1)
            self.b_in = nn.Parameter(torch.zeros(1, ))
            nn.init.kaiming_uniform_(self.W_out.weight, a=np.sqrt(5))

        if attention_type == 'concat':
            self.Wh = nn.Parameter(torch.randn(query_dim+key_dim, hidden_dim))
            self.Wa = nn.Parameter(torch.randn(hidden_dim, 1))
            self.ba = nn.Parameter(torch.zeros(1, ))
            nn.init.kaiming_uniform_(self.Wh, a=np.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=np.sqrt(5))

        self.dropout_rate = dropout
        if self.dropout_rate != None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        mask = None
        if len(inputs) == 2:
            query, key = inputs  # B * H; B * T * H
        elif len(inputs) == 3:
            query, key, mask = inputs  # B * H; B * T * H; B * T
        else:
            assert ValueError("Error in the size of FinalAttentionQKV")
        batch_size, time_step, input_dim = key.size()  # batch_size * input_dim + 1 * hidden_dim(i)
        input_q = self.W_q(query)  # B * h
        input_k = self.W_k(key)  # b t h
        if self.value_dim is None:
            input_v = key  # B * T * H
        else:
            input_v = self.W_v(key)  # b t h

        if self.attention_type == 'add':  # B*T*I  @ H*I
            q = torch.reshape(input_q, (batch_size, 1, self.hidden_dim))  # B*1*H
            h = q + input_k + self.b_in  # b t h
            h = self.tanh(h)  # B*T*H
            e = self.W_out(h)  # b t 1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        elif self.attention_type == 'mul':
            q = torch.reshape(input_q, (batch_size, self.hidden_dim, 1))  # B*h 1
            e = torch.matmul(input_k, q).squeeze()  # b t

        elif self.attention_type == 'concat':
            q = input_q.unsqueeze(1).repeat(1, time_step, 1)  # b t h
            k = input_k
            c = torch.cat((q, k), dim=-1)  # B*T*2I
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        if mask is not None:
            e = e - (1 - mask) * 10e6
        a = self.softmax(e)  # B*T
        if self.dropout is not None:
            a = self.dropout(a)
        v = torch.matmul(a.unsqueeze(1), input_v).squeeze()  # B * H
        return v, a, e  # B * H; B * T

class CohortExploitationModule(nn.Module):
    def __init__(self, f_num, f_dim, o_dim, cluster_num, attention_type="mul"):
        super(CohortExploitationModule, self).__init__()
        self.f_num = f_num
        self.f_dim = f_dim
        self.cluster_num = cluster_num

        self.active = nn.ReLU()
        self.cohort_attention = clones(FinalAttentionQKV(hidden_dim=f_dim, query_dim=f_dim,
                                                         key_dim=f_dim+1, value_dim=f_dim+1,
                                                         attention_type=attention_type), f_num)
        self.compress = nn.Linear(f_num * f_dim, f_dim)
        self.predictionCohort = nn.Linear((f_dim+1) * f_num, o_dim, bias=False)

    def forward(self, tdata, f_mask, cohorts):
        # tdata: B * T * F * Hs
        # f_mask: B * T * F
        batch_size, time_size, _, _ = tdata.shape
        fdata = torch.flatten(tdata, start_dim=0, end_dim=1)  # B*T * F * Cp
        # fdata = torch.flatten(tdata, start_dim=0, end_dim=1)  # B*T * F * H

        f_codes = None
        for i in range(self.f_num):
            mask = torch.flatten(f_mask[:, :, i], start_dim=0, end_dim=1).byte()  # B*T
            pos_f_codes = compute_codes(fdata[mask, i, :], cohorts[i]['centers'].cuda())  # B*T
            pos_f_codes = (pos_f_codes + 2).detach()
            mask_f_codes = torch.ones(fdata.shape[0], dtype=torch.long).cuda()
            sub_f_codes = torch.zeros(fdata.shape[0], dtype=torch.long).cuda()
            sub_f_codes = sub_f_codes.scatter_add_(0, torch.where(mask)[0], pos_f_codes)
            sub_f_codes = sub_f_codes.scatter_add_(0, torch.where(1 - mask)[0], mask_f_codes)
            if i == 0:
                f_codes = sub_f_codes.unsqueeze(1)
            else:
                f_codes = torch.cat((f_codes, sub_f_codes.unsqueeze(1)), 1).detach()

        cohort_a = []
        sample_f_cohort_rep = None
        feature_mask = torch.zeros((batch_size, self.f_num)).byte().cuda()
        for i in range(self.f_num):
            # select cohort pattern
            cohort_pattern = cohorts[i]['pat'].cuda()  # C * F
            cohort_pattern_rep = cohorts[i]['pat_rep'].cuda()  # C * H
            cohort_sample_pos_cnt = cohorts[i]['labeled']['pos_cnt_sample']  # C
            cohort_sample_neg_cnt = cohorts[i]['labeled']['neg_cnt_sample']  # C
            cohort_sample_pos_ratio = cohort_sample_pos_cnt/(cohort_sample_pos_cnt+cohort_sample_neg_cnt+1e-6)
            cohort_sample_pos_ratio = cohort_sample_pos_ratio.cuda()  # C
            cohort_pattern_rep = torch.cat((cohort_pattern_rep, cohort_sample_pos_ratio.unsqueeze(1)), 1)  # C * H+1
            cohort_size, feature_size = cohort_pattern.shape
            cohort_rep_size = cohort_pattern_rep.shape[1]

            # support more cohorts
            chunk = 1500
            pat_rep = torch.zeros((batch_size, cohort_size, cohort_rep_size)).cuda()
            pattern_match = torch.zeros((batch_size, cohort_size)).byte().cuda()
            for j in range(0, cohort_size, chunk):
                begin = j
                end = min(j + chunk, cohort_pattern.shape[0])
                sub_pattern = cohort_pattern[begin:end]
                sub_cohort_pat_rep = cohort_pattern_rep[begin:end]
                sub_pattern_mask = (sub_pattern != 0).byte()  # C * F
                pattern_code = f_codes.unsqueeze(1) * sub_pattern_mask  # B*T * C * F
                sub_pattern_match = (pattern_code == sub_pattern).all(2).byte()  # B*T * C
                sub_pattern_match_t = sub_pattern_match.reshape(-1, time_size, end - begin)  # B * T * C
                sub_pattern_match = sub_pattern_match_t.any(1).byte()  # B * C
                sub_pat_rep = sub_pattern_match.unsqueeze(2) * sub_cohort_pat_rep.unsqueeze(0)  # B * C * H
                pat_rep[:, begin:end, :] += sub_pat_rep
                pattern_match[:, begin:end] += sub_pattern_match

            subf_cohort_rep, subf_cohort_a, _ = self.cohort_attention[i]([tdata[:, -1, i, :], pat_rep, pattern_match])  # B * H, B * C
            cohort_a.append(subf_cohort_a)  # F * [B * C]
            feature_mask[:, i] += pattern_match.any(1).byte()

            if i == 0:
                sample_f_cohort_rep = subf_cohort_rep.unsqueeze(-2)  # B * 1 * H
            else:
                sample_f_cohort_rep = torch.cat((sample_f_cohort_rep, subf_cohort_rep.unsqueeze(-2)), 1)  # B * F * H

        sample_f_cohort_rep = sample_f_cohort_rep * f_mask.any(1).unsqueeze(2)  # B * F*H
        p_wc_rep = torch.flatten(sample_f_cohort_rep, start_dim=1, end_dim=2)  # B * F*H
        cohort_cali = self.predictionCohort(p_wc_rep)
        return cohort_cali, cohort_a, sample_f_cohort_rep


class CohortNet(nn.Module):
    def __init__(self, o_dim, f_num, e_dim, c_dim, h_dim, fusion_dim, cluster_num, clip_min, clip_max, active='relu'):
        super(CohortNet, self).__init__()
        self.o_dim = o_dim  # output dimension
        self.f_num = f_num  # feature dimension
        self.e_dim = e_dim  # embedding dimension
        self.c_dim = c_dim  # compression dimension
        self.h_dim = h_dim  # hidden dimension for RNN-based model
        self.fusion_dim = fusion_dim  # fusion dimension
        self.clip_min = clip_min  # clip_min for embedding
        self.clip_max = clip_max  # clip_max for embedding

        self.cohorts = None

        self.MFLM = MultiChannelFeatureLearningModule(f_num, e_dim, clip_max, clip_min, fusion_dim, h_dim,
                                                      c_dim, o_dim, active='relu')
        self.CEM = CohortExploitationModule(f_num, h_dim, o_dim, cluster_num)
        if o_dim == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(1)

    def forward(self, inputs):  # B * T * F  B * T * F
        tdata, tmask = inputs
        pre_out, t_out, f_mask, ff_a = self.MFLM(tdata, tmask)

        cohort_cali, cohort_a, sample_f_cohort_rep = None, None, None
        if self.cohorts != None:
            cohort_data = t_out
            cohort_cali, cohort_a, sample_f_cohort_rep = self.CEM(cohort_data, f_mask, self.cohorts)  # F * [B * C]; B * H

        if self.cohorts != None:
            pred = self.activation(pre_out + cohort_cali)
        else:
            pred = self.activation(pre_out)

        return pred, t_out, ff_a, cohort_a