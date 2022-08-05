import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('./data/test/images/16_test.tif').convert("RGB")
mask = Image.open('./data/test/1st_manual/16_manual1.gif').convert("L")
ANN_pred = Image.open('./ANN_test.png').convert("L")
INN_pred = Image.open('./INN_test.png').convert("L")


plt.figure()
plt.subplot(1, 4, 1)
plt.imshow(img)
plt.axis('off')
plt.title('image')
plt.subplot(1, 4, 2)
plt.imshow(mask, cmap='gray')
plt.axis('off')
plt.title('mask')
plt.subplot(1, 4, 3)
plt.imshow(ANN_pred, cmap='gray')
plt.axis('off')
plt.title('ANN result')
plt.subplot(1, 4, 4)
plt.imshow(INN_pred, cmap='gray')
plt.axis('off')
plt.title('INN result')
plt.savefig('./test_result_DRIVE.jpg', dpi=800)
plt.show()
