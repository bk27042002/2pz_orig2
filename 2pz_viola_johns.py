import cv2

# Загрузка классификатора лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Загрузка изображения и преобразование в оттенки серого
image = cv2.imread('image/4.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Обнаружение лиц на изображении
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Рисование прямоугольников вокруг обнаруженных лиц
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

# Отображение результата
image = cv2.resize(image, (551, 708))
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()