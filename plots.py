import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

# Box Loss Curve
plt.figure(figsize=(8,5))
plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
plt.plot(df['epoch'], df['val/box_loss'], label='Validation Box Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Box Loss Over Epochs')
plt.legend()
plt.tight_layout()
plt.savefig("box_loss_curve.png")
plt.close()

# mAP Curve
plt.figure(figsize=(8,5))
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.title('mAP Over Epochs')
plt.legend()
plt.tight_layout()
plt.savefig("map_curve.png")
plt.close()

# Precision–Recall Curve
plt.figure(figsize=(8,5))
plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Precision and Recall Over Epochs')
plt.legend()
plt.tight_layout()
plt.savefig("precision_recall_curve.png")
plt.close()

print("Graphs generated successfully!")
