There are a lot of files, don't worry, chillax.
The only files you need to touch really are config.py and main.py, from these files you can change all you need between different models, train them, save them, and evaluate them. The different files is because this is a lot of code and I though it be better to separate everything

## File Struct
On top of the CNN files you have to add the datasets files, both the dataset.py and the loaders.py and of course the dataset in this file org:

your_project/
├── datasets/
│   ├── raw/                          # Put your raw data here
│   │   ├── manifest.json            # Create this with the datasets
│   │   ├── subject1_trial1.npz
│   │   ├── subject1_trial2.npz
│   │   └── subject2_trial1.npz
│   │
│   └── processed/                    # Auto-generated (DO NOT TOUCH)
│       ├── train.pt
│       ├── val.pt
│       └── test.pt
│
├── models/                           # Your trained models
│   ├── nano_baseline.pth             # Example
│   └── resnet50_teacher.pth
│
├── config.py                         # ← EDIT THIS
├── main.py                    # ← RUN THIS
├── dataset.py                        
├── loaders.py                        
├── students.py
├── teachers.py
├── distillation.py
├── evaluation.py
├── training.py
└── utils.py

## USE INSTRUCTIONS
The only file you have to run is main.py. Before training a model, change in config.py what you need, the most important configs (model size, epochs, normal or knowledge distillation, etc...) are very visible. And finally if you want to change the name of the saved model files, you can change that in main.py by changing the very visible part.

## Extra
Again, I know these are crazy amount of filesm but at least there is zero brain involved. NGL I have been vibecoding this for some time, and I haven't tested anything, it's kinda time to go to bed, so if you're reading this, THIS IS NOT TESTED, DW I WILL TEST IT SOON, but I hope you see the vision.
