
import torch
import torch.nn as nn
from lib.pspnet import PSPNet
from lib.pointnet import Pointnet2MSG
from lib.adaptor import PriorAdaptor

from test_res_dgcnn import PR_Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        # print(f"MLP 输入形状 (转置前): {x.shape}")
        x = x.transpose(1, 2)  # 转置到 (bs, 1024, 128)
        # print(f"MLP 输入形状 (转置后): {x.shape}")
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.transpose(1, 2)  # 转置回 (bs, 128, 1024)
        # print(f"MLP 输出形状 (转置后): {x.shape}")
        return x

class CLGraspNet(nn.Module):
    def __init__(self, n_cat=6, nv_prior=1024, num_structure_points=128):
        super(CLGraspNet, self).__init__()
        self.n_cat = n_cat
        self.num_structure_points = num_structure_points

        self.instance_geometry = Pointnet2MSG(0)
        self.category_local = Pointnet2MSG(0)

        self.mlp1 = MLP(128, 128)  # 根据你的应用需求调整 MLP 维度
        self.mlp2 = MLP(320, 128)  # 根据你的应用需求调整 MLP 维度)
        conv1d_stpts_prob_modules = []
        conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1))
        conv1d_stpts_prob_modules.append(nn.LeakyReLU())
        conv1d_stpts_prob_modules.append(
            nn.Conv1d(in_channels=256, out_channels=self.num_structure_points, kernel_size=1))
        conv1d_stpts_prob_modules.append(nn.Softmax(dim=2))
        self.conv1d_stpts_prob = nn.Sequential(*conv1d_stpts_prob_modules)
        self.prior_enricher = PriorAdaptor(emb_dims=64, n_heads=4)
        self.assignment = nn.Sequential(
            nn.Conv1d(896, 896, 1),
            nn.LeakyReLU(),
            nn.Conv1d(896, 448, 1),
            nn.LeakyReLU(),
            nn.Conv1d(448, n_cat * nv_prior, 1),  # n_cat*nv_prior = 6*1024
        )
        self.deformation = nn.Sequential(
            nn.Conv1d(896, 896, 1),
            nn.LeakyReLU(),
            nn.Conv1d(896, 448, 1),
            nn.LeakyReLU(),
            nn.Conv1d(448, n_cat * 3, 1),  # n_cat*3 = 18
        )
        self.deformation[4].weight.data.normal_(0, 0.0001)

        self.layer_normal = nn.LayerNorm(1024)

        # self.Transformer = Transformer(3, 960, 1024, length=1024)

        self.psp = PSPNet(bins=(1, 2, 3, 6), backend='resnet18')
        self.instance_color = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.LeakyReLU(),
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
        )

        self.pr_model = PR_Model()





    def forward(self, points, img, choose, cat_id, prior):
        prior = self.pr_model(img, prior)
        input_points = points.clone()

        bs, n_pts = points.size()[:2]
        nv = prior.size()[1]  # 1024
        index = cat_id + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat
        points = self.instance_geometry(points)  # 形状 (bs, 64, 1024) 当前点云
        cat_points = self.category_local(prior)  # 形状 (bs, 64, 1024) 先验点云

        out_img = self.psp(img)
        di = out_img.size()[1]
        emb = out_img.view(bs, di, -1)
        choose = choose.unsqueeze(1).repeat(1, di, 1)  # bs x 1 x 1024 -> bs x 32 x 1024
        emb = torch.gather(emb, 2, choose).contiguous()
        emb = self.instance_color(emb)  # bs x 64 x 1024

        ins_cate = torch.cat([points, emb], dim=1)  # 形状 (bs, 128, 1024) 当前点云


        embedding = self.prior_enricher(cat_points, points, emb)

        ins_cate_1 = self.mlp1(ins_cate) # 当前点云

        cate_ins_1 = self.mlp1(embedding) # 先验点云


        new_cate_ins = torch.cat([points,ins_cate_1,cate_ins_1], dim=1)  # 形状 (bs, 384, 1024) x_1 当前点云
        new_ins_cate = torch.cat([cat_points,cate_ins_1,ins_cate_1], dim=1)  # 形状 (bs, 384, 1024) x_3 先验点云
        m= new_cate_ins
        n =new_ins_cate
        diff =new_cate_ins-new_ins_cate

        new_cate_ins = self.mlp2(new_cate_ins) # 映射矩阵

        new_ins_cate = self.mlp2(new_ins_cate) # 变形矩阵

        new_cate_ins  = self.pool(new_cate_ins)  # bs x 128 x 1
        diff = self.pool(diff)  # bs x 128 x 1
        new_ins_cate = self.pool(new_ins_cate)  # bs x 128 x 1



        # assignemnt matrix
        assign_feat=torch.cat((m, new_cate_ins.repeat(1, 1, n_pts), diff.repeat(1, 1, n_pts), new_ins_cate.repeat(1, 1, n_pts)),dim=1)
        assign_mat = self.assignment(assign_feat)  # bs x (6*1024) x n_pts # 映射矩阵
        assign_mat = assign_mat.view(-1,nv,n_pts).contiguous()  # bs, nc*nv, n_pts ->f bs*nc, nv, n_pts (bs*6 x 1024 x n_pts)
        assign_mat = torch.index_select(assign_mat, 0, index)  # bs x nv x n_pts
        assign_mat = assign_mat.permute(0, 2, 1).contiguous()  # bs x n_pts x nv

        # deformation field
        deform_feat=torch.cat((n, new_ins_cate.repeat(1, 1, nv), diff.repeat(1, 1, n_pts), new_cate_ins.repeat(1, 1, nv)), dim = 1)
        deltas = self.deformation(deform_feat) # bs x (6*3) x n_pts
        deltas = deltas.view(-1, 3, nv).contiguous()  # bs, nc*3, nv -> bs*nc, 3, nv  (bs*6 x 3 x n_pts)
        deltas = torch.index_select(deltas, 0, index)  # bs x 3 x 1024
        deltas = deltas.permute(0, 2, 1).contiguous()

        new_shape = self.conv1d_stpts_prob(ins_cate)
        new_shape = torch.sum(new_shape[:, :, :, None] * input_points[:, None, :, :], dim=2)

        return new_shape, assign_mat, deltas


