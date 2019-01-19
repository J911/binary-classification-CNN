# Jaemin Lee (aka, J911)
# 2019

import trainer
import detector

train_dataset, test_dataset = trainer.make_dataset()
train_loader, test_loader = trainer.get_train_loader(train_dataset), trainer.get_train_loader(test_dataset)

model_path = './model_save'

for epoch in range(1, 10):
    trainer.train(epoch, train_loader)
    trainer.test(test_loader)

trainer.save_model(model_path)

detector.load_model(model_path)
detector.start_detect()