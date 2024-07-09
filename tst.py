from deepforest import main
import cv2
import numpy as np




img_path = "img_1.png"
img=cv2.imread(img_path)

# Convert the image to the expected format
img_array = np.array(img).astype('float32')



model= main.deepforest()
model.use_release()



# Predict trees in the image
box_info = model.predict_image(img_array)
print(box_info)




# Draw circles around detected trees
for n in range(len(box_info)):
    x = (box_info.iloc[n]['xmin'] + box_info.iloc[n]['xmax']) / 2
    y = (box_info.iloc[n]['ymin'] + box_info.iloc[n]['ymax']) / 2
    cv2.circle(img, (int(x), int(y)), 25, (0, 255, 0), 2)

# Add text to the image
cv2.putText(img, 'TOTAL TREES: ' + str(len(box_info)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Display the output image
cv2.imshow('output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()