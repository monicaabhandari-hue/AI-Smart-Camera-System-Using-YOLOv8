from ultralytics import YOLO

# Loading the trained model
model = YOLO("my_yolo_model/weights/best.pt")

#detecting on the test folder
model.predict(source="C:/Users/monica/Desktop/test", save=True)

print("Done")
