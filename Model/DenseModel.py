import pytorch_lightning as pl
import torch
from torch import nn

label_key = 'MGMT_value'
modalities = 'T1w', 'T1wCE', 'T2w', 'FLAIR'

class DenseModel(pl.LightningModule):

    def __init__(self, input_size=(251, 251, 150)):
        super().__init__()

        # images are (1, 251, 251, 150) (channels, width, height, depth)
        height, width, depth = input_size

        dim = 8
        self.layer_1 = nn.Linear(height * width * depth, 8)
        self.layer_2 = nn.Linear(8, 256)
        self.layer_3 = nn.Linear(256, 1)

    def forward(self, x):

        # print("row", row)
        x = x.float()
        # print("X", x.size())
        # print("x", x, x.size(), torch.max(x), torch.mean(x), torch.min(x))


        batch, channels, height, width, depth = x.size()

        # (b, 1, 251, 251, 150) -> (b, 1*251*251*150)
        x = x.view(batch, -1)
        x = self.layer_1(x)
        x = nn.functional.relu(x)
        x = self.layer_2(x)
        x = nn.functional.relu(x)
        x = self.layer_3(x)

        return torch.relu(torch.squeeze(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        row = batch
        y = row[label_key].float()
        x = row['T2w']['data']
        y_hat = self(x)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # print("yaht", y_hat)
        # print("y", y)
        return loss_fn(y_hat, y)

    def test_step(self, batch, batch_idx):
        row = batch
        y = row[label_key].float()
        x = row['T2w']['data']
        y_hat = self(x)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # print("yaht", y_hat)
        # print("y", y)
        return loss_fn(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


if __name__ == '__main__':

    # model = DenseModel()
    # input = torch.randn(251*251*150)
    # print("1", input)
    # input = input.view(1, 1, 251, 251, 150)
    # print("2", input)

    # output = model(input)
    # print(output)

    import sys
    sys.path.append('../Dataset/')
    from RSNAPre import RSNAPre

    input_size = (64, 64, 64)
    batch_size = 32

    dataset = RSNAPre(
        data_dir='/Volumes/GoogleDrive/我的雲端硬碟/KaggleBrain/rsna-preprocessed', batch_size = batch_size, input_size = input_size)
    dataset.setup()

    loader = dataset.train_dataloader()

    dummyX = torch.rand(batch_size*input_size[0]*input_size[1]*input_size[2])
    dummyX = dummyX.view(batch_size, 1, *input_size)
    model = DenseModel(input_size=input_size)
    model(dummyX)

    trainer = pl.Trainer()
    trainer.fit(model, dataset)
