# from hezar.models import Model
#
# example = ["هزار، کتابخانه‌ای کامل برای به کارگیری آسان هوش مصنوعی"]
# model = Model.load("hezarai/bert-fa-sentiment-dksf")
# outputs = model.predict(example)
# print(outputs)


from hezar.models import Model
# OCR with TrOCR
model = Model.load("hezarai/trocr-base-fa-v2")
texts = model.predict(["a.jpg"])
print(f"TrOCR Output: {texts}")

# OCR with CRNN
model = Model.load("hezarai/crnn-base-fa-64x256")
texts = model.predict("a.jpg")
print(f"CRNN Output: {texts}")