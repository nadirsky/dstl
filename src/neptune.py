from deepsense import neptune
from PIL import Image
import time

class Neptune():
    def __init__(self):
        ctx = neptune.Context()

        self.logs_channel = ctx.job.create_channel(
            name='logs',
            channel_type=neptune.ChannelType.TEXT)

        self.epoch_train_loss_channel = ctx.job.create_channel(
            name='epoch_train_loss',
            channel_type=neptune.ChannelType.NUMERIC)
        self.epoch_train_acc_channel = ctx.job.create_channel(
            name='epoch_train_acc',
            channel_type=neptune.ChannelType.NUMERIC)

        self.epoch_val_loss_channel = ctx.job.create_channel(
            name='epoch_val_loss',
            channel_type=neptune.ChannelType.NUMERIC)
        self.epoch_val_acc_channel = ctx.job.create_channel(
            name='epoch_val_acc',
            channel_type=neptune.ChannelType.NUMERIC)

        self.image_channel = ctx.job.create_channel(
            name='false_predictions',
            channel_type=neptune.ChannelType.IMAGE)

        ctx.job.create_chart(
            name='Epoch loss',
            series={
                'training': self.epoch_train_loss_channel,
                'validation': self.epoch_val_loss_channel
            }
        )

        ctx.job.create_chart(
            name='Epoch accuracy',
            series={
                'training': self.epoch_train_acc_channel,
                'validation': self.epoch_val_acc_channel
            }
        )

    def send_neptune_image(self, raw_image, name, description):
        image = Image.fromarray(raw_image)
        neptune_image = neptune.Image(
            name=name,
            description=description,
            data=image)
        self.image_channel.send(x=time.time(), y=neptune_image)

    def print_to_neptune(self, message):
        self.logs_channel.send(x=time.time(), y=message)