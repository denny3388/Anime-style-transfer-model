import time, itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
from glob import glob
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import random

''' remove 'module.' of DataParallel/DistributedDataParallel '''
def removeModel(dictionary):
    remove = False
    for k in dictionary.keys():
        if k[0:7] == 'module.':
            remove = True
            break
    if remove:
        new_state_dict = OrderedDict()
        for k, v in dictionary.items():
            name = k[7:] 
            new_state_dict[name] = v
        return new_state_dict
    return dictionary

''' turn 'label1'2'label2' into string '''
def labelTranslate(label1, label2):
    label1, label2 = label1.to('cpu'), label2.to('cpu')
    if torch.equal(label1, torch.tensor([[1, 0, 0, 0]])): # F_young
        str1 = 'FY'
    elif torch.equal(label1, torch.tensor([[0, 1, 0, 0]])): # F_middle
        str1 = 'FM'
    elif torch.equal(label1, torch.tensor([[0, 0, 1, 0]])): # M_young
        str1 = 'MY'
    elif torch.equal(label1, torch.tensor([[0, 0, 0, 1]])): # M_middle
        str1 = 'MM'

    if torch.equal(label2, torch.tensor([[1, 0, 0, 0]])): # F_young
        str2 = 'FY'
    elif torch.equal(label2, torch.tensor([[0, 1, 0, 0]])): # F_middle
        str2 = 'FM'
    elif torch.equal(label2, torch.tensor([[0, 0, 1, 0]])): # M_young
        str2 = 'MY'
    elif torch.equal(label2, torch.tensor([[0, 0, 0, 1]])): # M_middle
        str2 = 'MM'
    
    return str1 + '2' + str2

def str2labelTensor(string):
    if 'FY' in string:
        return torch.tensor([[1, 0, 0, 0]])
    elif 'FM' in string:
        return torch.tensor([[0, 1, 0, 0]])
    elif 'MY' in string:
        return torch.tensor([[0, 0, 1, 0]])
    elif 'MM' in string:
        return torch.tensor([[0, 0, 0, 1]])
    else:
        print('invalid input!')
        return 0


class UGATIT(object) :
    def __init__(self, args):
        self.light = args.light

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.result_dir = args.result_dir
        self.dataset = args.dataset

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        """ DDP """
        self.DDP = args.DDP
        self.rank = args.local_rank

        """ Writer """
        self.exp_name = args.exp_name

        """ Number of each label """
        self.F_Y = 0
        self.F_M = 0
        self.M_Y = 0
        self.M_M = 0

        # Initialize DDP
        if self.DDP:
            torch.distributed.init_process_group(backend="nccl", rank=self.rank, world_size=3) # world size = number of GPU
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)
            self.local_rank = local_rank

        # Initialize writer
        self.writer = SummaryWriter('loss/' + self.exp_name)

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)
        print("# experiment name : ", self.exp_name)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):

        """ DataLoader """
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.img_size + 30, self.img_size+30)),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.trainA = ImageFolder(os.path.join('dataset', self.dataset, 'trainA'), train_transform)
        '''
        self.trainB = ImageFolder(os.path.join('dataset', self.dataset, 'trainB'), train_transform)
        '''
        self.trainB_M_Y = ImageFolder(os.path.join('dataset', self.dataset, 'trainB_M_Y'), train_transform)
        self.trainB_M_M = ImageFolder(os.path.join('dataset', self.dataset, 'trainB_M_M'), train_transform)
        self.trainB_F_Y = ImageFolder(os.path.join('dataset', self.dataset, 'trainB_F_Y'), train_transform)
        self.trainB_F_M = ImageFolder(os.path.join('dataset', self.dataset, 'trainB_F_M'), train_transform)
        self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'testA'), test_transform)
        self.testB = ImageFolder(os.path.join('dataset', self.dataset, 'testB'), test_transform)

        # for DDP
        if self.DDP:
            sampler=DistributedSampler(self.trainA) 
            self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, sampler=sampler)
            sampler=DistributedSampler(self.trainB_M_Y)
            self.trainB_M_Y_loader = DataLoader(self.trainB_M_Y, batch_size=self.batch_size, sampler=sampler)
            sampler=DistributedSampler(self.trainB_M_M)
            self.trainB_M_M_loader = DataLoader(self.trainB_M_M, batch_size=self.batch_size, sampler=sampler)
            sampler=DistributedSampler(self.trainB_F_Y)
            self.trainB_F_Y_loader = DataLoader(self.trainB_F_Y, batch_size=self.batch_size, sampler=sampler)
            sampler=DistributedSampler(self.trainB_F_M)
            self.trainB_F_M_loader = DataLoader(self.trainB_F_M, batch_size=self.batch_size, sampler=sampler)
            self.trainB_loaders = [self.trainB_F_Y_loader, self.trainB_F_M_loader, self.trainB_M_Y_loader, self.trainB_M_M_loader]
            self.iters = [0,0,0,0]

            self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
            self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False)
        else: # No DDP
            self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True)
            self.trainB_M_Y_loader = DataLoader(self.trainB_M_Y, batch_size=self.batch_size, shuffle=True)
            self.trainB_M_M_loader = DataLoader(self.trainB_M_M, batch_size=self.batch_size, shuffle=True)
            self.trainB_F_Y_loader = DataLoader(self.trainB_F_Y, batch_size=self.batch_size, shuffle=True)
            self.trainB_F_M_loader = DataLoader(self.trainB_F_M, batch_size=self.batch_size, shuffle=True)
            self.trainB_loaders = [self.trainB_F_Y_loader, self.trainB_F_M_loader, self.trainB_M_Y_loader, self.trainB_M_M_loader]
            self.iters = [0,0,0,0]

            self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
            self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False)

        """ Define Generator, Discriminator, add label """
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        """ Trainer """
        self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = RhoClipper(0, 1)

    def train(self):
        self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()
        
        start_iter = 1
        if self.resume:
            model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.dataset, 'model'), start_iter)
                print(" [*] Load SUCCESS")
                if self.decay_flag and start_iter > (self.iteration // 2):
                    self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                    self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)

        # DDP
        if self.DDP:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.genA2B = torch.nn.parallel.DistributedDataParallel(self.genA2B, device_ids=[self.local_rank], output_device=self.local_rank)
            self.genB2A = torch.nn.parallel.DistributedDataParallel(self.genB2A, device_ids=[self.local_rank], output_device=self.local_rank)
            self.disGA = torch.nn.parallel.DistributedDataParallel(self.disGA, device_ids=[self.local_rank], output_device=self.local_rank)
            self.disGB = torch.nn.parallel.DistributedDataParallel(self.disGB, device_ids=[self.local_rank], output_device=self.local_rank)
            self.disLA = torch.nn.parallel.DistributedDataParallel(self.disLA, device_ids=[self.local_rank], output_device=self.local_rank)
            self.disLB = torch.nn.parallel.DistributedDataParallel(self.disLB, device_ids=[self.local_rank], output_device=self.local_rank)

        # training loop
        print('training start !')
        start_time = time.time()
        for step in range(start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            # randomly choose a image from real faces
            try:
                real_A, label_A = trainA_iter.next()
            except:
                trainA_iter = iter(self.trainA_loader)
                real_A, label_A = trainA_iter.next()
            
            # randomly choose a label from anime face (label_B)
            index = random.randint(0,3)

            # choose the corresponding anime face
            try:
                real_B, label_B = self.iters[index].next()
            except:
                self.iters[index] = iter(self.trainB_loaders[index])
                real_B, label_B = self.iters[index].next()
            ####################################################
            
            real_A, real_B = real_A.to(self.device), real_B.to(self.device)

            # Update D
            self.D_optim.zero_grad()

            fake_A2B, _, _ = self.genA2B(real_A, label_A, label_B)
            fake_B2A, _, _ = self.genB2A(real_B, label_B, label_A)

            real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A, label_A)
            real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A, label_A)
            real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B, label_B)
            real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B, label_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A, label_A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A, label_A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B, label_B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B, label_B)

            ''' discriminate real images and fake images as more as possible, smaller loss, greater performance '''
            D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
            D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(self.device)) + self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
            D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit).to(self.device)) + self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
            D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(self.device)) + self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
            D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit).to(self.device)) + self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

            Discriminator_loss = D_loss_A + D_loss_B
            if step % 2 == 0: # update D every 2 steps
                Discriminator_loss.backward()
                self.D_optim.step()
            
            # record D_loss
            self.writer.add_scalar('D_loss', Discriminator_loss, step) 
            self.writer.add_scalar('D_loss_A', D_loss_A, step) 
            self.writer.add_scalar('D_loss_B', D_loss_B, step) 

            if torch.equal(label_A, torch.tensor([[1, 0, 0, 0]])): # F_young
                self.writer.add_scalar('D_loss_F_Y', Discriminator_loss, step)
            elif torch.equal(label_A, torch.tensor([[0, 1, 0, 0]])): # F_middle
                self.writer.add_scalar('D_loss_F_M', Discriminator_loss, step)
            elif torch.equal(label_A, torch.tensor([[0, 0, 1, 0]])): # M_young
                self.writer.add_scalar('D_loss_M_Y', Discriminator_loss, step)
            elif torch.equal(label_A, torch.tensor([[0, 0, 0, 1]])): # M_middle
                self.writer.add_scalar('D_loss_M_M', Discriminator_loss, step)
            
            # Update G
            self.G_optim.zero_grad()

            fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A, label_A, label_B)
            fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B, label_B, label_A)

            fake_A2B2A, _, _ = self.genB2A(fake_A2B, label_B, label_A)
            fake_B2A2B, _, _ = self.genA2B(fake_B2A, label_A, label_B)

            fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A, label_B, label_A)
            fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B, label_A, label_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A, label_A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A, label_A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B, label_B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B, label_B)

            ''' let fake images to be like real images '''
            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
            G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
            G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device))
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
            G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
            G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device))

            ''' cycle loss '''
            G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

            G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
            G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

            G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit).to(self.device)) + self.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.device))
            G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(self.device)) + self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device))

            G_loss_A =  self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
            G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B

            Generator_loss = G_loss_A + G_loss_B
            Generator_loss.backward()
            self.G_optim.step()

            # record G_loss
            self.writer.add_scalar('G_loss', Generator_loss, step)

            self.writer.add_scalar('G_loss_A', G_loss_A, step)
            self.writer.add_scalar('G_adv_loss_A', G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA, step)
            self.writer.add_scalar('G_cycle_loss_A', G_recon_loss_A, step)
            self.writer.add_scalar('G_identity_loss_A', G_identity_loss_A, step)
            self.writer.add_scalar('G_cam_loss_A', G_cam_loss_A, step)

            self.writer.add_scalar('G_loss_B', G_loss_B, step)
            self.writer.add_scalar('G_adv_loss_B', G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB, step)
            self.writer.add_scalar('G_cycle_loss_B', G_recon_loss_B, step)
            self.writer.add_scalar('G_identity_loss_B', G_identity_loss_B, step)
            self.writer.add_scalar('G_cam_loss_B', G_cam_loss_B, step)

            if torch.equal(label_A, torch.tensor([[1, 0, 0, 0]])): # F_young
                self.writer.add_scalar('G_loss_F_Y', Generator_loss, step)
            elif torch.equal(label_A, torch.tensor([[0, 1, 0, 0]])): # F_middle
                self.writer.add_scalar('G_loss_F_M', Generator_loss, step)
            elif torch.equal(label_A, torch.tensor([[0, 0, 1, 0]])): # M_young
                self.writer.add_scalar('G_loss_M_Y', Generator_loss, step)
            elif torch.equal(label_A, torch.tensor([[0, 0, 0, 1]])): # M_middle
                self.writer.add_scalar('G_loss_M_M', Generator_loss, step)

            # clip parameter of AdaILN and ILN, applied after optimizer step
            self.genA2B.apply(self.Rho_clipper)
            self.genB2A.apply(self.Rho_clipper)

            print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f, rank: %d" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss, self.rank))
            if step % self.print_freq == 0:
                train_sample_num = 5
                test_sample_num = 5
                A2B = np.zeros((self.img_size * 7, 0, 3))
                B2A = np.zeros((self.img_size * 7, 0, 3))

                self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()
                for _ in range(train_sample_num):

                    # randomly choose a image from real faces
                    try:
                        real_A, label_A = trainA_iter.next()
                    except:
                        trainA_iter = iter(self.trainA_loader)
                        real_A, label_A = trainA_iter.next()

                    # randomly choose a label from anime face (label_B)
                    index = random.randint(0,3)

                    # choose the corresponding anime face
                    try:
                        real_B, label_B = self.iters[index].next()
                    except:
                        self.iters[index] = iter(self.trainB_loaders[index])
                        real_B, label_B = self.iters[index].next()

                    real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                    # add label as input of generator and discriminator
                    fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A, label_A, label_B)
                    fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B, label_B, label_A)

                    fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B, label_B, label_A)
                    fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A, label_A, label_B)

                    fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A, label_B, label_A)
                    fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B, label_A, label_B)
                    ####################################################

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                               cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                               cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                               cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                               cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                               cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                for _ in range(test_sample_num):
                    # randomly choose a image from real faces
                    try:
                        real_A, label_A = trainA_iter.next()
                    except:
                        trainA_iter = iter(self.trainA_loader)
                        real_A, label_A = trainA_iter.next()

                    # randomly choose a label from anime face (label_B)
                    index = random.randint(0,3)

                    # choose the corresponding anime face
                    try:
                        real_B, label_B = self.iters[index].next()
                    except:
                        self.iters[index] = iter(self.trainB_loaders[index])
                        real_B, label_B = self.iters[index].next()

                    real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                    # add label as input of generator and discriminator
                    fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A, label_A, label_B)
                    fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B, label_B, label_A)

                    fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B, label_B, label_A)
                    fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A, label_A, label_B)

                    fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A, label_B, label_A)
                    fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B, label_A, label_B)
                    ####################################################

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                               cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                               cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                               cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                               cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                               cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                labelA2B, labelB2A = labelTranslate(label_A, label_B), labelTranslate(label_B, label_A)
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d_%s.png' % (step, labelA2B)), A2B * 255.0)
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d_%s.png' % (step, labelB2A)), B2A * 255.0)
                self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

            if step % self.save_freq == 0 and self.rank == 0:
                self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)

            if step % 1000 == 0 and self.rank == 0:
                print('saved latest model! (%d)' % step)
                params = {}
                params['genA2B'] = self.genA2B.state_dict()
                params['genB2A'] = self.genB2A.state_dict()
                params['disGA'] = self.disGA.state_dict()
                params['disGB'] = self.disGB.state_dict()
                params['disLA'] = self.disLA.state_dict()
                params['disLB'] = self.disLB.state_dict()
                torch.save(params, os.path.join(self.result_dir, self.dataset + '_params_latest.pt'))

    def save(self, dir, step):
        print('save model! (%d)' % step)
        params = {}
        params['genA2B'] = self.genA2B.state_dict()
        params['genB2A'] = self.genB2A.state_dict()
        params['disGA'] = self.disGA.state_dict()
        params['disGB'] = self.disGB.state_dict()
        params['disLA'] = self.disLA.state_dict()
        params['disLB'] = self.disLB.state_dict()
        torch.save(params, os.path.join(dir, self.dataset + '_params_%07d.pt' % step))

    def load(self, dir, step):
        params = torch.load(os.path.join(dir, self.dataset + '_params_%07d.pt' % step), map_location=str(self.device))
        self.genA2B.load_state_dict(removeModel(params['genA2B']))
        self.genB2A.load_state_dict(removeModel(params['genB2A']))
        self.disGA.load_state_dict(removeModel(params['disGA']))
        self.disGB.load_state_dict(removeModel(params['disGB']))
        self.disLA.load_state_dict(removeModel(params['disLA']))
        self.disLB.load_state_dict(removeModel(params['disLB']))

    def test(self):
        model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
        if not len(model_list) == 0:
            model_list.sort()
            iter = int(model_list[-1].split('_')[-1].split('.')[0])
            self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return

        self.genA2B.eval(), self.genB2A.eval()
        for n, (real_A, label_A) in enumerate(self.testA_loader):
            print('processing ', n, 'th image... (A->B)')
            label_B = str2labelTensor('MM') # set test anime label
            real_A, label_A, label_B = real_A.to(self.device), label_A.to(self.device), label_B.to(self.device)

            # add label as input of generator and discriminator
            fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A, label_A, label_B)

            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B, label_B, label_A)

            fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A, label_B, label_A)
            ####################################################

            A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                  cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                  cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                  cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)

            labelA2B = labelTranslate(label_A, label_B)
            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'A2B_%d_%s.png' % ((n + 1), labelA2B)), A2B * 255.0)

        for n, (real_B, label_B) in enumerate(self.testB_loader):
            print('processing ', n, 'th image... (B->A)')
            label_A = label_B
            real_B, label_B, label_A = real_B.to(self.device), label_B.to(self.device), label_A.to(self.device)

            fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B, label_B, label_A)

            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A, label_A, label_B)

            fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B, label_A, label_B)

            B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                  cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                  cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                  cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)

            labelB2A = labelTranslate(label_B, label_A)
            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'B2A_%d_%s.png' % ((n + 1), labelB2A)), B2A * 255.0)
