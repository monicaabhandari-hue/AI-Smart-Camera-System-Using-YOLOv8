import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

# Boss Loss
plt.figure(figsize=(10,5))
plt.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss", linewidth=2)
plt.plot(df["epoch"], df["val/box_loss"], label="Validation Box Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Box Loss Over Epochs")
plt.legend()
plt.tight_layout()
plt.savefig("box_loss_curve.png", dpi=300, bbox_inches="tight")
plt.show()

# Classification Loss
plt.figure(figsize=(10,5))
plt.plot(df["epoch"], df["train/cls_loss"], label="Train Classification Loss", color="orange", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Classification Loss Over Epochs")
plt.legend()
plt.tight_layout()
plt.savefig("classification_loss_curve.png", dpi=300, bbox_inches="tight")
plt.show()

#DFL Loss
plt.figure(figsize=(10,5))
plt.plot(df["epoch"], df["train/dfl_loss"], label="Train DFL Loss", color="green", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DFL Loss Over Epochs")
plt.legend()
plt.tight_layout()
plt.savefig("dfl_loss_curve.png", dpi=300, bbox_inches="tight")
plt.show()

