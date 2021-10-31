import os
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import InpaintGenerator, Discriminator
from .loss import AdversarialLoss


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0
        self.epoch = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']
            self.epoch = data['epoch']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])


    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'epoch': self.epoch,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)



class InpaintingModel(BaseModel):
    def __init__(self, config, append_name=''):
        super(InpaintingModel, self).__init__('InpaintingModel'+str(append_name), config)

        # generator input: [rgb(3)]
        # discriminator input: [rgb(3)]
        generator = InpaintGenerator(in_channels=config.INPUT_CHANNELS)

        discriminator = Discriminator(in_channels=config.INPUT_CHANNELS, use_sigmoid=config.GAN_LOSS != 'hinge')


        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)

        l1_loss = nn.MSELoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)


        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )


    def process(self, images, masks=None):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs = self(images, masks)
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real, dis_real_feat = self.discriminator(dis_input_real)                    # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)                    # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # generator l1 loss
        if masks is not None:
            gen_l1_loss_back = self.l1_loss(outputs*(1-masks).float(), images*(1-masks).float())
            gen_l1_loss_crop = self.l1_loss(outputs*masks.float(), images*masks.float())
            gen_l1_loss = (gen_l1_loss_back + 4 * gen_l1_loss_crop) * self.config.REC_LOSS_WEIGHT / torch.mean(masks)
        else:
            gen_l1_loss = self.l1_loss(outputs, images) * self.config.REC_LOSS_WEIGHT
        gen_loss += gen_l1_loss

        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += nn.MSELoss()(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss

        logs = {
            "l_dis": dis_loss.item(),
            "l_dis_real": dis_real_loss.item(),
            "l_dis_fake": dis_fake_loss.item(),
            "l_gen_gan": gen_gan_loss.item(),
            "l_l1": gen_l1_loss.item(),
            "l_fm": gen_fm_loss.item(),
            "l_gen_sum": gen_loss.item(),
        }

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, masks=None):
        if masks is not None:
            images_masked = (images * (1 - masks).float()) + masks
            inputs = images_masked
            if masks.size(0) == 1:
                masks = masks.expand(inputs.size()[0], -1, -1, -1)
            outputs = self.generator(inputs, masks)  # in: [rgb(3)]
        else:
            inputs = images
            outputs = self.generator(inputs)                                    # in: [rgb(3)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()

