import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
from torch.autograd import Variable
import numpy as np
import pdb
if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.DIMENSION * cfg.VIDEO_LEN
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.relu = nn.ReLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=True):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if bcondition:
            self.outlogits = nn.Sequential(
                conv3x3(ndf * 8 + nef, ndf * 8),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())
        else:
            self.outlogits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        # conditioning output    
        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)


# ############# Networks for stageI GAN #############
class StoryGAN(nn.Module):
    def __init__(self, video_len):
        super(StoryGAN, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM * 8
        self.motion_dim = cfg.TEXT.DIMENSION + cfg.LABEL_NUM
        self.content_dim = cfg.GAN.CONDITION_DIM # encoded text dim
        self.noise_dim = cfg.GAN.Z_DIM  # noise
        self.recurrent = nn.GRUCell(self.noise_dim + self.motion_dim, self.motion_dim)
        self.mocornn = nn.GRUCell(self.motion_dim, self.content_dim)
        self.video_len = video_len
        self.n_channels = 3
        self.filter_num = 3
        self.filter_size = 21
        self.image_size = 124
        self.out_num = 1
        self.define_module()

    def define_module(self):
        from layers import DynamicFilterLayer1D as DynamicFilterLayer
        ninput = self.motion_dim + self.content_dim + self.image_size
        ngf = self.gf_dim

        self.ca_net = CA_NET()
        # -> ngf x 4 x 4
        self.fc = nn.Sequential(
            nn.Linear(ninput, ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4),
            nn.ReLU(True))

        self.filter_net = nn.Sequential(
            nn.Linear(self.content_dim, self.filter_size * self.filter_num * self.out_num),
            nn.BatchNorm1d(self.filter_size * self.filter_num * self.out_num))

        self.image_net = nn.Sequential(
            nn.Linear(self.motion_dim, self.image_size * self.filter_num),
            nn.BatchNorm1d(self.image_size * self.filter_num),
            nn.Tanh())

        # ngf x 4 x 4 -> ngf/2 x 8 x 8
        self.upsample1 = upBlock(ngf, ngf // 2)
        # -> ngf/4 x 16 x 16
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        # -> ngf/8 x 32 x 32
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        # -> ngf/16 x 64 x 64
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        # -> 3 x 64 x 64
        self.img = nn.Sequential(
            conv3x3(ngf // 16, 3),
            nn.Tanh())

        self.m_net = nn.Sequential(
            nn.Linear(self.motion_dim, self.motion_dim),
            nn.BatchNorm1d(self.motion_dim))

        self.c_net = nn.Sequential(
            nn.Linear(self.content_dim, self.content_dim),
            nn.BatchNorm1d(self.content_dim))

        self.dfn_layer = DynamicFilterLayer(self.filter_size, 
            pad = self.filter_size//2)

    def get_iteration_input(self, motion_input):
        num_samples = motion_input.shape[0]
        noise = T.FloatTensor(num_samples, self.noise_dim).normal_(0,1)
        return torch.cat((noise, motion_input), dim = 1)

    def get_gru_initial_state(self, num_samples):
        return Variable(T.FloatTensor(num_samples, self.motion_dim).normal_(0, 1))

    def sample_z_motion(self, motion_input, video_len=None):
        video_len = video_len if video_len is not None else self.video_len
        num_samples = motion_input.shape[0]
        h_t = [self.m_net(self.get_gru_initial_state(num_samples))]
        
        for frame_num in range(video_len):
            if len(motion_input.shape) == 2:
                e_t = self.get_iteration_input(motion_input)
            else:
                e_t = self.get_iteration_input(motion_input[:,frame_num,:])
            h_t.append(self.recurrent(e_t, h_t[-1]))
        z_m_t = [h_k.view(-1, 1, self.motion_dim) for h_k in h_t]
        z_motion = torch.cat(z_m_t[1:], dim=1).view(-1, self.motion_dim)
        return z_motion

    def motion_content_rnn(self, motion_input, content_input):
        video_len = 1 if len(motion_input.shape) == 2 else self.video_len
        h_t = [self.c_net(content_input)]
        if len(motion_input.shape) == 2:
            motion_input = motion_input.unsqueeze(1)
        for frame_num in range(video_len):
            h_t.append(self.mocornn(motion_input[:,frame_num, :], h_t[-1]))
        
        c_m_t = [h_k.view(-1, 1, self.content_dim) for h_k in h_t]
        mocornn_co = torch.cat(c_m_t[1:], dim=1).view(-1, self.content_dim)
        return mocornn_co

    def sample_videos(self, motion_input, content_input):  
        content_input = content_input.view(-1, cfg.VIDEO_LEN * content_input.shape[2])
        r_code, r_mu, r_logvar = self.ca_net(torch.squeeze(content_input))
        c_code = r_code.repeat(self.video_len, 1).view(-1, r_code.shape[1])
        c_mu = r_mu.repeat(self.video_len, 1).view(-1, r_mu.shape[1])
        c_logvar = r_logvar.repeat(self.video_len, 1).view(-1, r_logvar.shape[1])

        crnn_code = self.motion_content_rnn(motion_input, r_code)
        
        temp = motion_input.view(-1, motion_input.shape[2])
        m_code, m_mu, m_logvar = temp, temp, temp #self.ca_net(temp)
        m_code = m_code.view(motion_input.shape[0], self.video_len, self.motion_dim)
        zm_code = self.sample_z_motion(m_code, self.video_len)

        # one
        zmc_code = torch.cat((zm_code, c_mu), dim = 1)
        # two
        m_image = self.image_net(m_code.view(-1, m_code.shape[2]))
        m_image = m_image.view(-1, self.filter_num, self.image_size)
        c_filter = self.filter_net(crnn_code)
        c_filter = c_filter.view(-1, self.out_num, self.filter_num, self.filter_size)
        mc_image = self.dfn_layer([m_image, c_filter])
        zmc_all = torch.cat((zmc_code, mc_image.squeeze(1)), dim = 1)
        #combine
        zmc_all = self.fc(zmc_all)
        zmc_all = zmc_all.view(-1, self.gf_dim, 4, 4)
        h_code = self.upsample1(zmc_all)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)
        # state size 3 x 64 x 64
        h = self.img(h_code)
        fake_video = h.view(h.size(0) / self.video_len, self.video_len, self.n_channels, h.size(3), h.size(3))
        fake_video = fake_video.permute(0, 2, 1, 3, 4)
        return None, fake_video,  m_mu, m_logvar, r_mu, r_logvar

    def sample_images(self, motion_input, content_input):  
        m_code, m_mu, m_logvar = motion_input, motion_input, motion_input #self.ca_net(motion_input)
        content_input = content_input.view(-1, cfg.VIDEO_LEN * content_input.shape[2])
        c_code, c_mu, c_logvar = self.ca_net(content_input)
        crnn_code = self.motion_content_rnn(motion_input, c_mu)
        zm_code = self.sample_z_motion(m_code, 1)
        # one
        zmc_code = torch.cat((zm_code, c_mu), dim = 1)
        # two
        m_image = self.image_net(m_code)
        m_image = m_image.view(-1, self.filter_num, self.image_size)
        c_filter = self.filter_net(crnn_code)
        c_filter = c_filter.view(-1, self.out_num, self.filter_num, self.filter_size)
        mc_image = self.dfn_layer([m_image, c_filter])
        zmc_all = torch.cat((zmc_code, mc_image.squeeze(1)), dim = 1)
        #combine
        zmc_all = self.fc(zmc_all)
        zmc_all = zmc_all.view(-1, self.gf_dim, 4, 4)
        h_code = self.upsample1(zmc_all)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)
        # state size 3 x 64 x 64
        fake_img = self.img(h_code)
        return None, fake_img, m_mu, m_logvar, c_mu, c_logvar

class STAGE1_D_IMG(nn.Module):
    def __init__(self, use_categories = True):
        super(STAGE1_D_IMG, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.text_dim = cfg.TEXT.DIMENSION
        self.label_num = cfg.LABEL_NUM
        self.define_module(use_categories)

    def define_module(self, use_categories):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            # state size (ndf * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.get_cond_logits = D_GET_LOGITS(ndf, nef + self.text_dim + self.label_num)
        self.get_uncond_logits = None

        if use_categories:
            self.cate_classify = nn.Conv2d(ndf * 8, self.label_num, 4, 4, 1, bias = False)
        else:
            self.cate_classify = None

    def forward(self, image):
        img_embedding = self.encode_img(image)

        return img_embedding


class STAGE1_D_STY_V2(nn.Module):
    def __init__(self):
        super(STAGE1_D_STY_V2, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.text_dim = cfg.TEXT.DIMENSION
        self.label_num = cfg.LABEL_NUM
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            # state size (ndf * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.get_cond_logits = D_GET_LOGITS(ndf, nef + self.text_dim + self.label_num)
        self.get_uncond_logits = None
        self.cate_classify = None

    def forward(self, story):
        N, C, video_len, W, H = story.shape
        story = story.permute(0,2,1,3,4)
        story = story.contiguous().view(-1, C,W,H)
        story_embedding = torch.squeeze(self.encode_img(story))
        _, C1, W1, H1 = story_embedding.shape
        story_embedding = story_embedding.view(N,video_len, C1, W1, H1)
        story_embedding = story_embedding.mean(1).squeeze()
        return story_embedding


# ############# Networks for stageII GAN #############
class STAGE2_G(nn.Module):
    def __init__(self, STAGE1_G, video_len = 5):
        super(STAGE2_G, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM
        self.motion_dim = cfg.TEXT.DIMENSION + cfg.LABEL_NUM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.n_channels = 3
        self.z_dim = cfg.Z_DIM
        self.video_len = video_len
        self.STAGE1_G = STAGE1_G
        # fix parameters of stageI GAN
        for param in self.STAGE1_G.parameters():
            param.requires_grad = False
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        self.ca_net = CA_NET()
        # --> 4ngf x 16 x 16
        self.encoder = nn.Sequential(
            conv3x3(3, ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True))
        self.hr_joint = nn.Sequential(
            conv3x3(self.ef_dim + self.motion_dim + ngf * 4, ngf * 4),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True))
        self.residual = self._make_layer(ResBlock, ngf * 4)
        # --> 2ngf x 32 x 32
        self.upsample1 = upBlock(ngf * 4, ngf * 2)
        # --> ngf x 64 x 64
        self.upsample2 = upBlock(ngf * 2, ngf)
        # --> ngf // 2 x 128 x 128
        self.upsample3 = upBlock(ngf, ngf // 2)
        # --> 3 x 128 x 128
        self.img = nn.Sequential(
            conv3x3(ngf // 2, 3),
            nn.Tanh())


    def sample_images(self, motion_input, content_input):
        _, stage1_img, _, _, _, _ = self.STAGE1_G.sample_images(motion_input, content_input)
        stage1_img = stage1_img.detach()
        encoded_img = self.encoder(stage1_img)
        ## Text
        m_code, m_mu, m_logvar = motion_input, motion_input, motion_input #self.ca_net(motion_input)
        content_input = content_input.view(-1, cfg.VIDEO_LEN * content_input.shape[2])
        c_code, c_mu, c_logvar = self.ca_net(content_input)
        mc_code = torch.cat([m_code, c_code], 1)

        mc_code = mc_code.view(-1, self.ef_dim + self.motion_dim, 1, 1)
        mc_code = mc_code.repeat(1, 1, 16, 16)

        i_c_code = torch.cat([encoded_img, mc_code], 1)
        h_code = self.hr_joint(i_c_code)
        h_code = self.residual(h_code)

        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)

        fake_img = self.img(h_code)

        return stage1_img, fake_img, m_mu, m_logvar, c_mu, c_logvar

    def sample_videos(self, motion_input, content_input):
        _, stage1_story, _, _, _, _ = self.STAGE1_G.sample_videos(motion_input, content_input)
        stage1_img = stage1_story.permute(0,2,1,3,4)
        stage1_img = stage1_img.contiguous().view(-1, stage1_img.shape[2], stage1_img.shape[3], stage1_img.shape[4])
        stage1_img = stage1_img.detach()
        encoded_img = self.encoder(stage1_img)
        ## Text
        temp = motion_input.view(-1, motion_input.shape[2])
        m_code, m_mu, m_logvar = temp, temp, temp #self.ca_net(temp)
        content_input = content_input.view(-1, cfg.VIDEO_LEN * content_input.shape[2])
        r_code, r_mu, r_logvar = self.ca_net(content_input)

        c_code = r_code.repeat(self.video_len, 1).view(-1, r_code.shape[1])
        c_mu = r_mu.repeat(self.video_len, 1).view(-1, r_mu.shape[1])
        c_logvar = r_logvar.repeat(self.video_len, 1).view(-1, r_logvar.shape[1])

        mc_code = torch.cat((m_code, c_code), dim = 1)

        mc_code = mc_code.view(-1, self.ef_dim + self.motion_dim, 1, 1)
        mc_code = mc_code.repeat(1, 1, 16, 16)

        i_c_code = torch.cat([encoded_img, mc_code], 1)
        h_code = self.hr_joint(i_c_code)
        h_code = self.residual(h_code)

        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)

        h = self.img(h_code)
        fake_video = h.view(h.size(0) / self.video_len, self.video_len, self.n_channels, h.size(3), h.size(3))
        fake_video = fake_video.permute(0, 2, 1, 3, 4)
        return stage1_story, fake_video, m_mu, m_logvar, r_mu, r_logvar


class STAGE2_D_IMG(nn.Module):
    def __init__(self, use_categories = True):
        super(STAGE2_D_IMG, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.text_dim = cfg.TEXT.DIMENSION
        self.label_num = cfg.LABEL_NUM
        self.define_module(use_categories)

    def define_module(self, use_categories):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),  # 64 * 64 * ndf
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),  # 32 * 32 * ndf * 2
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),  # 16 * 16 * ndf * 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # 8 * 8 * ndf * 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),  # 4 * 4 * ndf * 16
            conv3x3(ndf * 16, ndf * 16),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),   # 4 * 4 * ndf * 16
            conv3x3(ndf * 16, ndf * 8),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)   # 4 * 4 * ndf * 8
        )

        self.get_cond_logits = D_GET_LOGITS(ndf, nef + self.text_dim + self.label_num, bcondition=True)
        self.get_uncond_logits = D_GET_LOGITS(ndf, nef + self.text_dim + self.label_num, bcondition=False)
        if use_categories:
            self.cate_classify = nn.Conv2d(ndf * 8, self.label_num, 4, 4, 1, bias = False)
        else:
            self.cate_classify = None

    def forward(self, image):
        img_embedding = self.encode_img(image)

        return img_embedding


class STAGE2_D_STY(nn.Module):
    def __init__(self):
        super(STAGE2_D_STY, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.text_dim = cfg.TEXT.DIMENSION
        self.label_num = cfg.LABEL_NUM
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),  # 64 * 64 * ndf
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),  # 32 * 32 * ndf * 2
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),  # 16 * 16 * ndf * 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # 8 * 8 * ndf * 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),  # 4 * 4 * ndf * 16
            conv3x3(ndf * 16, ndf * 16),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),   # 4 * 4 * ndf * 16
            conv3x3(ndf * 16, ndf * 8),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)   # 4 * 4 * ndf * 8
        )

        self.get_cond_logits = D_GET_LOGITS(ndf, nef + self.text_dim + self.label_num, bcondition=True)
        self.get_uncond_logits = D_GET_LOGITS(ndf, nef + self.text_dim + self.label_num, bcondition=False)
        self.cate_classify = None

    def forward(self, story):
        N, C, video_len, W, H = story.shape
        story = story.permute(0,2,1,3,4)
        story = story.contiguous().view(-1, C,W,H)
        story_embedding = torch.squeeze(self.encode_img(story))
        _, C1, W1, H1 = story_embedding.shape
        story_embedding = story_embedding.view(N,video_len, C1, W1, H1)
        story_embedding = story_embedding.mean(1).squeeze()
        return story_embedding


