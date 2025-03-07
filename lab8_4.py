import cv2
import numpy as np


# Функция для обнаружения метки и наложения изображения мухи
def track_marker(frame, fly_img):
    # Преобразуем изображение в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Применяем Гауссово размытие для уменьшения шума
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Детектор кругов Хафа
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=100, param2=30, minRadius=10,
                               maxRadius=100)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Рисуем круг
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            # Наложение изображения мухи на кадр
            fly_resized = cv2.resize(fly_img, (r * 2, r * 2))  # Изменяем размер изображения мухи по радиусу метки
            fly_center = (
            x - fly_resized.shape[1] // 2, y - fly_resized.shape[0] // 2)  # Центр мухи совпадает с центром метки

            # Проверяем, чтобы изображение мухи не выходило за пределы кадра
            fly_center_x = max(0, fly_center[0])
            fly_center_y = max(0, fly_center[1])

            # Определяем область на кадре, куда будем накладывать изображение
            end_x = fly_center_x + fly_resized.shape[1]
            end_y = fly_center_y + fly_resized.shape[0]

            # Проверяем, что изображения помещаются в кадр
            if end_x <= frame.shape[1] and end_y <= frame.shape[0]:
                roi = frame[fly_center_y:end_y, fly_center_x:end_x]

                # Проверка на наличие альфа-канала у изображения мухи
                if fly_resized.shape[2] == 4:  # Если изображение имеет альфа-канал
                    fly_gray = cv2.cvtColor(fly_resized, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(fly_gray, 1, 255, cv2.THRESH_BINARY)

                    # Применяем маску для наложения
                    fly_masked = cv2.bitwise_and(fly_resized, fly_resized, mask=mask)
                    roi_masked = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))

                    # Применяем выравнивание по размерам (создаем копии, чтобы избежать ошибок)
                    roi_masked_resized = cv2.resize(roi_masked, (fly_resized.shape[1], fly_resized.shape[0]))
                    fly_masked_resized = cv2.resize(fly_masked, (fly_resized.shape[1], fly_resized.shape[0]))

                    # Объединяем оригинальный кадр с изображением мухи
                    frame[fly_center_y:end_y, fly_center_x:end_x] = cv2.add(roi_masked_resized, fly_masked_resized)
                else:
                    # Если альфа-канала нет, просто заменяем участок кадра на муху
                    frame[fly_center_y:end_y, fly_center_x:end_x] = fly_resized

    return frame


# Загружаем изображение мухи с альфа-каналом (если есть)
fly_img = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)  # Если изображение с альфа-каналом, используем IMREAD_UNCHANGED

# Подключаемся к камере
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Не удалось захватить изображение")
        break

    # Обрабатываем кадр и накладываем изображение мухи
    processed_frame = track_marker(frame, fly_img)

    # Отображаем результат
    cv2.imshow('Marker Tracking with Fly', processed_frame)

    # Прерываем цикл при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закрываем все окна
cap.release()
cv2.destroyAllWindows()



