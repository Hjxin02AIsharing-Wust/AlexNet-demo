import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from Net import AlexNet
import os
import json
import time
from Data_Preprocess import Data_Preprocess
class Train:
    def __init__(self):
        pass
    def train(self,device):


        # data loader
        root_file='D:/github/datasets/AlexNet_demo_dataset'
        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(227),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            "val": transforms.Compose([transforms.Resize((227, 227)),  # cannot 224, must (224, 224)
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
        train_dataset = datasets.ImageFolder(root=root_file + "/train",
                                             transform=data_transform["train"])
        train_num = len(train_dataset)
        flower_list = train_dataset.class_to_idx
        cla_dict = dict((val, key) for key, val in flower_list.items())
        json_str = json.dumps(cla_dict, indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)
        batch_size = 128
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, shuffle=True,
                                                   num_workers=0)
        validate_dataset = datasets.ImageFolder(root=root_file + "/val",
                                                transform=data_transform["val"])
        val_num = len(validate_dataset)
        validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                      batch_size=batch_size, shuffle=True,
                                                      num_workers=0)

        test_data_iter = iter(validate_loader)
        test_image, test_label = test_data_iter.next()
        print(test_image[0].size(),type(test_image[0]))
        print(test_label[0],test_label[0].item(),type(test_label[0]))
        print("data loader done!")
        net = AlexNet(num_classes=5, init_weights=True)
        net.to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.0002)
        save_path = './AlexNet.pth'
        best_acc = 0.0
        #train epoch=10
        for epoch in range(10):
            # train
            net.train()    #During training, the dropout from the previously defined network is used
            running_loss = 0.0
            t1 = time.perf_counter()
            for step, data in enumerate(train_loader, start=0):
                images, labels = data
                optimizer.zero_grad()
                outputs = net(images.to(device))
                loss = loss_function(outputs, labels.to(device))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                # print train process
                rate = (step + 1) / len(train_loader)
                a = "*" * int(rate * 50)
                b = "." * int((1 - rate) * 50)
                print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
            print()
            print(time.perf_counter() - t1)
            # validate
            net.eval()  # No dropout during testing, using all neurons
            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                for val_data in validate_loader:
                    val_images, val_labels = val_data
                    outputs = net(val_images.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += (predict_y == val_labels.to(device)).sum().item()
                val_accurate = acc / val_num
                if val_accurate > best_acc:
                    best_acc = val_accurate
                    torch.save(net.state_dict(), save_path)
                print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, running_loss / step, val_accurate))

        print('Finished Training')
if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Data_Preprocess=Data_Preprocess()
    Data_Preprocess.run()
    Train=Train()
    Train.train(device)