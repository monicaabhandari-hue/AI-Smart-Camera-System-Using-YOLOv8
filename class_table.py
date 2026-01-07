import os
import pandas as pd

# COCO class names
class_names = ['aeroplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove',
'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli',
'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'cow', 'cup',
'diningtable', 'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe',
'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife', 'laptop',
'microwave', 'motorbike', 'mouse', 'orange', 'oven', 'parking meter', 'person', 'pizza',
'pottedplant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink',
'skateboard', 'skis', 'snowboard', 'sofa', 'spoon', 'sports ball', 'stop sign',
'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet',
'toothbrush', 'traffic light', 'train', 'truck', 'tvmonitor', 'umbrella', 'vase',
'wine glass', 'zebra']

# Path where YOLO saved labels (train + valid)
labels_train = r"C:\Users\monica\Desktop\Capstone\datasets\train\labels"
labels_valid = r"C:\Users\monica\Desktop\Capstone\datasets\valid\labels"

# Count array for all 80 classes
counts = [0] * len(class_names)

def count_labels(folder):
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r") as f:
                for line in f:
                    cls = int(line.split()[0])
                    counts[cls] += 1

# Count both training and validation labels
count_labels(labels_train)
count_labels(labels_valid)

# Create dataframe
df = pd.DataFrame({"Class": class_names, "Count": counts})

# Sort by count
df_sorted = df.sort_values(by="Count", ascending=False)


df_sorted.to_csv("class_distribution_table.csv", index=False)



